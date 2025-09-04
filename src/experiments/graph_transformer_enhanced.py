import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from config.graph_transformer_config_enhanced import parse_args


if __name__ == "__main__":
    args = parse_args()
    MODEL_NAME = args.model_name
    SUFFIX = args.data_suffix
    DATASET = args.dataset
    USE_POSITIONAL_ENCODING = not args.disable_pos_encoding
    
    import optuna
    from optuna.trial import TrialState
    import torch
    from models.GraphTransformer import GraphTransformer, GraphTransformerWithPositionalEncoding
    from utils.graph_transformer_utils_enhanced import (
        set_seed,
        get_metrics,
        load_enhanced_data,
        load_multiple_enhanced_data,
        create_enhanced_model_with_dataset_config,
        create_enhanced_model_from_config,
        enhanced_train,
        enhanced_evaluate,
        save_enhanced_model,
        load_enhanced_model,
        create_enhanced_model_from_checkpoint,
        DEVICE,
    )
    from tqdm import tqdm
    import gc
    from torch.utils.tensorboard import SummaryWriter
    import json

    set_seed(args.seed)
    
    print(f"Enhanced Graph Transformer Experiment")
    print(f"=====================================")
    print(f"Using dataset: {DATASET}")
    print(f"Data directory: {args.data_dir}")
    print(f"Data suffix: {SUFFIX}")
    print(f"Positional encodings: {'Enabled' if USE_POSITIONAL_ENCODING else 'Disabled'}")
    print(f"Model name: {MODEL_NAME}")
    
    # Initialize tensorboard writer if not disabled
    writer = None
    if not args.disable_tensorboard:
        log_dir = os.path.join(args.log_dir, MODEL_NAME)
        writer = SummaryWriter(log_dir=log_dir)
        print(f"Tensorboard logging enabled: {log_dir}")
    
    def create_enhanced_objective(train_dataloader, val_dataloader, num_epochs, writer=None):
        """Create objective function for Optuna optimization with positional encoding support."""
        
        def objective(trial):
            # Suggest hyperparameters
            embedding_dim = trial.suggest_categorical('embedding_dim', [128, 256, 512])
            hidden_dim = trial.suggest_categorical('hidden_dim', [64, 128, 256])
            num_heads = trial.suggest_categorical('num_heads', [4, 8, 16])
            num_layers = trial.suggest_categorical('num_layers', [1, 2, 3, 4])
            dropout = trial.suggest_float('dropout', 0.0, 0.3)
            learning_rate = trial.suggest_float('learning_rate', 1e-4, 1e-2, log=True)
            pooling_method = trial.suggest_categorical('pooling_method', ['mean', 'max', 'add'])
            
            # Positional encoding hyperparameters (if enabled)
            pos_config = {}
            if USE_POSITIONAL_ENCODING and args.optimize_pos_encoding:
                # Get dataset limits
                if hasattr(train_dataloader, 'dataset_obj') and hasattr(train_dataloader.dataset_obj, 'get_positional_encoding_config'):
                    dataset_config = train_dataloader.dataset_obj.get_positional_encoding_config(use_percentile=True)
                    if dataset_config:
                        max_dataset_depth = dataset_config['max_depth']
                        max_dataset_child = dataset_config['max_child_index']
                        
                        # Suggest within reasonable ranges based on dataset
                        pos_config['depth_embedding_dim'] = trial.suggest_categorical('depth_embedding_dim', [16, 32, 64])
                        pos_config['child_embedding_dim'] = trial.suggest_categorical('child_embedding_dim', [16, 32, 64])
                        pos_config['max_depth'] = min(max_dataset_depth, trial.suggest_int('max_depth_limit', max_dataset_depth//2, max_dataset_depth))
                        pos_config['max_child_index'] = min(max_dataset_child, trial.suggest_int('max_child_index_limit', max_dataset_child//2, max_dataset_child))
            
            # Create model configuration
            config = {
                'embedding_dim': embedding_dim,
                'hidden_dim': hidden_dim,
                'num_heads': num_heads,
                'num_layers': num_layers,
                'dropout': dropout,
                'pooling_method': pooling_method,
                'learning_rate': learning_rate
            }
            config.update(pos_config)
            
            # Create model
            model = create_enhanced_model_from_config(
                num_node_features=train_dataloader.num_node_features,
                config=config,
                model_name=f"{MODEL_NAME}_trial_{trial.number}",
                use_positional_encoding=USE_POSITIONAL_ENCODING
            )
            
            optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3, factor=0.5)
            
            # Training
            criterion = torch.nn.BCEWithLogitsLoss()
            metrics = get_metrics()
            
            # Check if dataset has positional encodings
            has_pos_encodings = False
            if hasattr(train_dataloader, 'dataset_obj') and hasattr(train_dataloader.dataset_obj, 'has_positional_encodings'):
                has_pos_encodings = train_dataloader.dataset_obj.has_positional_encodings()
            
            best_val_loss = float('inf')
            
            for epoch in range(num_epochs):
                # Training phase
                model.train()
                for batch in train_dataloader:
                    batch = batch.to(DEVICE)
                    optimizer.zero_grad()
                    
                    # Forward pass with optional positional encodings
                    if USE_POSITIONAL_ENCODING and isinstance(model, GraphTransformerWithPositionalEncoding):
                        node_depth = getattr(batch, 'node_depth', None) if has_pos_encodings else None
                        child_index = getattr(batch, 'child_index', None) if has_pos_encodings else None
                        outputs = model(x=batch.x, edge_index=batch.edge_index, batch=batch.batch,
                                       node_depth=node_depth, child_index=child_index)
                    else:
                        outputs = model(x=batch.x, edge_index=batch.edge_index, batch=batch.batch)
                    
                    loss = criterion(outputs.squeeze(), batch.y.float())
                    loss.backward()
                    optimizer.step()
                
                # Validation phase
                val_loss, val_metrics = enhanced_evaluate(
                    model, val_dataloader, criterion, metrics, USE_POSITIONAL_ENCODING, has_pos_encodings
                )
                
                scheduler.step(val_loss)
                best_val_loss = min(best_val_loss, val_loss)
                
                # Report intermediate value to Optuna
                trial.report(val_loss, epoch)
                
                # Pruning
                if trial.should_prune():
                    raise optuna.exceptions.TrialPruned()
            
            return best_val_loss
            
        return objective
    
    if args.eval:
        # Evaluation mode only - load model and evaluate on test set
        print("Running in evaluation mode...")
        
        # Load test data
        test_dataloader = load_enhanced_data(
            data_dir=args.data_dir,
            split="test", 
            shuffle=False,
            batch_size=args.batch_size,
            suffix=SUFFIX,
            dataset=DATASET,
        )
        
        # Load model directly from checkpoint (includes architecture info)
        checkpoint_path = f"models/graph_transformer_enhanced/{MODEL_NAME}/{MODEL_NAME}_best.pth"
        
        try:
            model, optimizer, scheduler, epoch, best_vloss, best_vacc = create_enhanced_model_from_checkpoint(
                checkpoint_path, model_name=MODEL_NAME
            )
            print(f"Loaded model from epoch {epoch}")
            print(f"Best validation loss: {best_vloss:.4f}, Best validation accuracy: {best_vacc:.4f}")
        except FileNotFoundError:
            print(f"Error: No saved model found for {MODEL_NAME}")
            print(f"Expected path: {checkpoint_path}")
            sys.exit(1)
        except ValueError as e:
            print(f"Error: {e}")
            print("This checkpoint was saved with an older version that doesn't include architecture information.")
            print("Please re-train the model or use the migration script.")
            sys.exit(1)
        except Exception as e:
            print(f"Error loading model: {e}")
            sys.exit(1)
        
        # Check model type and dataset compatibility
        is_pos_model = isinstance(model, GraphTransformerWithPositionalEncoding)
        has_pos_data = hasattr(test_dataloader, 'dataset_obj') and hasattr(test_dataloader.dataset_obj, 'has_positional_encodings') and test_dataloader.dataset_obj.has_positional_encodings()
        
        print(f"Model uses positional encodings: {is_pos_model}")
        print(f"Dataset has positional encodings: {has_pos_data}")
        
        if is_pos_model and not has_pos_data:
            print("Warning: Model expects positional encodings but dataset doesn't provide them")
            print("Model will use zero-padding for missing positional data")
        
        # Evaluate on test set
        criterion = torch.nn.BCEWithLogitsLoss()
        metrics = get_metrics()
        test_loss, test_metrics = enhanced_evaluate(
            model, test_dataloader, criterion, metrics, is_pos_model, has_pos_data
        )

        # Log evaluation results to tensorboard
        if writer is not None:
            writer.add_scalar("eval/test_loss", test_loss, epoch)
            for metric_name, metric_value in test_metrics.items():
                writer.add_scalar(f"eval/test_{metric_name.lower()}", metric_value, epoch)
            writer.close()

        print("\n" + "=" * 50)
        print("ENHANCED MODEL EVALUATION RESULTS:")
        print("=" * 50)
        print(f"Test Loss: {test_loss:.5f}")
        for metric_name, metric_value in test_metrics.items():
            print(f"Test {metric_name}: {metric_value:.6f}")
        print("=" * 50)
        
    else:
        # Training or optimization mode
        train_loader, val_loader = load_multiple_enhanced_data(
            data_dir=args.data_dir,
            batch_size=args.batch_size,
            suffix=SUFFIX,
            dataset=DATASET,
        )

        if args.optimize:
            print("Running enhanced hyperparameter optimization...")
            os.makedirs("optuna", exist_ok=True)
            study = optuna.create_study(
                storage=args.storage_url, 
                study_name=args.study_name,
                direction="minimize", 
                load_if_exists=True
            )
            
            objective = create_enhanced_objective(
                train_dataloader=train_loader,
                val_dataloader=val_loader,
                num_epochs=args.epochs,
                writer=writer,
            )
            
            study.optimize(objective, n_trials=args.n_trials)

            # Log optimization results
            if writer is not None:
                writer.add_scalar("optuna/best_value", study.best_trial.value, len(study.trials))
                for key, value in study.best_trial.params.items():
                    writer.add_scalar(f"optuna/best_params/{key}", value, len(study.trials))
                writer.close()

            pruned_trials = study.get_trials(deepcopy=False, states=[TrialState.PRUNED])
            complete_trials = study.get_trials(deepcopy=False, states=[TrialState.COMPLETE])

            print("Enhanced optimization statistics: ")
            print("  Number of finished trials: ", len(study.trials))
            print("  Number of pruned trials: ", len(pruned_trials))
            print("  Number of complete trials: ", len(complete_trials))

            print("Best trial:")
            print(f"  Value: {study.best_trial.value}")
            print("  Params:")
            for key, value in study.best_trial.params.items():
                print(f"    {key}: {value}")

        elif args.train:
            print("Training enhanced model...")
            
            # Create model configuration
            config = {
                'embedding_dim': args.embedding_dim,
                'hidden_dim': args.hidden_dim,
                'num_heads': args.num_heads,
                'num_layers': args.num_layers,
                'dropout': args.dropout,
                'pooling_method': args.pooling_method,
                'learning_rate': args.learning_rate
            }
            
            # Add positional encoding config if not disabled
            if USE_POSITIONAL_ENCODING:
                pos_config = {
                    'depth_embedding_dim': args.depth_embed_dim,
                    'child_embedding_dim': args.child_embed_dim
                }
                
                # Use manual settings if provided, otherwise use dataset config
                if args.max_depth is not None:
                    pos_config['max_depth'] = args.max_depth
                if args.max_child_index is not None:
                    pos_config['max_child_index'] = args.max_child_index
                    
                config.update(pos_config)
            
            if args.use_best_params:
                print("Loading best hyperparameters from Optuna study...")
                # TODO: Implement Optuna parameter loading for enhanced version
                # For now, use the provided parameters
                print("Note: Optuna parameter loading for enhanced version not yet implemented")
                print("Using provided/default parameters")
            
            # Create model with dataset-derived configuration
            model, optimizer, scheduler = create_enhanced_model_with_dataset_config(
                dataloader=train_loader,
                model_name=MODEL_NAME,
                custom_config=config,
                use_positional_encoding=USE_POSITIONAL_ENCODING
            )
            
            criterion = torch.nn.BCEWithLogitsLoss()
            metrics = get_metrics()
            
            # Check dataset positional encoding availability
            has_pos_encodings = False
            if hasattr(train_loader, 'dataset_obj') and hasattr(train_loader.dataset_obj, 'has_positional_encodings'):
                has_pos_encodings = train_loader.dataset_obj.has_positional_encodings()
            
            print(f"Dataset has positional encodings: {has_pos_encodings}")
            
            enhanced_train(
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                criterion=criterion,
                train_dataloader=train_loader,
                val_dataloader=val_loader,
                metrics=metrics,
                num_epochs=args.epochs,
                writer=writer,
                use_positional_encoding=USE_POSITIONAL_ENCODING
            )

            # Clean up RAM to make room for the evaluation data
            del train_loader, val_loader
            gc.collect()

            # Final evaluation on test set
            test_dataloader = load_enhanced_data(
                data_dir=args.data_dir,
                split="test", 
                shuffle=False,
                batch_size=args.batch_size,
                suffix=SUFFIX,
                dataset=DATASET,
            )

            epoch, best_vloss, best_vacc = load_enhanced_model(
                model, optimizer, save_path="models/graph_transformer_enhanced", scheduler=scheduler
            )
            
            # Check test dataset positional encoding availability
            test_has_pos_encodings = False
            if hasattr(test_dataloader, 'dataset_obj') and hasattr(test_dataloader.dataset_obj, 'has_positional_encodings'):
                test_has_pos_encodings = test_dataloader.dataset_obj.has_positional_encodings()
            
            test_loss, test_metrics = enhanced_evaluate(
                model, test_dataloader, criterion, metrics, USE_POSITIONAL_ENCODING, test_has_pos_encodings
            )

            # Log final test results
            if writer is not None:
                writer.add_scalar("final/test_loss", test_loss, epoch)
                for metric_name, metric_value in test_metrics.items():
                    writer.add_scalar(f"final/test_{metric_name.lower()}", metric_value, epoch)
                writer.close()

            print("\n" + "=" * 50)
            print("ENHANCED MODEL - FINAL TEST RESULTS:")
            print("=" * 50)
            print(f"Test Loss: {test_loss:.5f}")
            for metric_name, metric_value in test_metrics.items():
                print(f"Test {metric_name}: {metric_value:.6f}")
            print("=" * 50)

        elif args.resume:
            print("Resuming enhanced model training...")
            
            # Load model directly from checkpoint (includes architecture info)
            checkpoint_path = f"models/graph_transformer_enhanced/{MODEL_NAME}/{MODEL_NAME}_best.pth"
            
            try:
                model, optimizer, scheduler, start_epoch, best_vloss, best_vacc = create_enhanced_model_from_checkpoint(
                    checkpoint_path, model_name=MODEL_NAME
                )
                print(f"Resuming from epoch {start_epoch}")
                print(f"Best validation loss so far: {best_vloss:.4f}, Best validation accuracy: {best_vacc:.4f}")
            except FileNotFoundError:
                print(f"Error: No saved model found for {MODEL_NAME}")
                print(f"Expected path: {checkpoint_path}")
                print("Use --train mode to train a new model")
                sys.exit(1)
            except ValueError as e:
                print(f"Error: {e}")
                print("This checkpoint was saved with an older version that doesn't include architecture information.")
                print("Please re-train the model or use the migration script.")
                sys.exit(1)
            except Exception as e:
                print(f"Error loading model: {e}")
                sys.exit(1)
            
            # Determine if model uses positional encodings
            model_uses_pos_encoding = isinstance(model, GraphTransformerWithPositionalEncoding)
            print(f"Loaded model uses positional encodings: {model_uses_pos_encoding}")
            
            # Respect the original scheduler configuration from checkpoint
            if scheduler is None:
                print("No scheduler was used in original training, continuing without scheduler...")
                
            criterion = torch.nn.BCEWithLogitsLoss()
            metrics = get_metrics()
            
            # Check dataset positional encoding availability
            has_pos_encodings = False
            if hasattr(train_loader, 'dataset_obj') and hasattr(train_loader.dataset_obj, 'has_positional_encodings'):
                has_pos_encodings = train_loader.dataset_obj.has_positional_encodings()
            
            # Continue training for additional epochs
            print(f"Continuing training for {args.epochs} more epochs...")
            enhanced_train(
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                criterion=criterion,
                train_dataloader=train_loader,
                val_dataloader=val_loader,
                metrics=metrics,
                num_epochs=args.epochs,
                initial_best_vloss=best_vloss,
                initial_best_vacc=best_vacc,
                writer=writer,
                start_epoch=start_epoch,
                use_positional_encoding=model_uses_pos_encoding
            )

            # Clean up RAM to make room for the evaluation data
            del train_loader, val_loader
            gc.collect()

            # Final evaluation on test set
            test_dataloader = load_enhanced_data(
                data_dir=args.data_dir,
                split="test", 
                shuffle=False,
                batch_size=args.batch_size,
                suffix=SUFFIX,
                dataset=DATASET,
            )

            epoch, best_vloss, best_vacc = load_enhanced_model(
                model, optimizer, save_path="models/graph_transformer_enhanced", scheduler=scheduler
            )
            
            test_has_pos_encodings = False
            if hasattr(test_dataloader, 'dataset_obj') and hasattr(test_dataloader.dataset_obj, 'has_positional_encodings'):
                test_has_pos_encodings = test_dataloader.dataset_obj.has_positional_encodings()
            
            test_loss, test_metrics = enhanced_evaluate(
                model, test_dataloader, criterion, metrics, model_uses_pos_encoding, test_has_pos_encodings
            )

            # Log resumed training final test results
            if writer is not None:
                writer.add_scalar("resumed/test_loss", test_loss, epoch)
                for metric_name, metric_value in test_metrics.items():
                    writer.add_scalar(f"resumed/test_{metric_name.lower()}", metric_value, epoch)
                writer.close()

            print("\n" + "=" * 50)
            print("ENHANCED MODEL - RESUMED TRAINING FINAL TEST RESULTS:")
            print("=" * 50)
            print(f"Test Loss: {test_loss:.5f}")
            for metric_name, metric_value in test_metrics.items():
                print(f"Test {metric_name}: {metric_value:.6f}")
            print("=" * 50)
