import sys
import os
import gc
import torch
import optuna

# Add parent directory to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from config.gcn_config_enhanced import parse_args


if __name__ == "__main__":
    args = parse_args()
    MODEL_NAME = args.model_name
    SUFFIX = args.data_suffix
    DATASET = args.dataset
    USE_POSITIONAL_ENCODING = not args.disable_pos_encoding
    
    import torch
    import torch.nn as nn
    import optuna
    from optuna.trial import TrialState
    
    from models.GCN import GCN, GCNWithPositionalEncoding
    from utils.gcn_utils_enhanced import (
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
    
    set_seed(args.seed)
    
    print(f"Enhanced GCN Experiment")
    print(f"======================")
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
            # Suggest core hyperparameters
            embedding_dim = trial.suggest_categorical('embedding_dim', [128, 256, 512])
            hidden_dim_1 = trial.suggest_categorical('hidden_dim_1', [64, 128, 256])
            hidden_dim_2 = trial.suggest_categorical('hidden_dim_2', [64, 128, 256])
            dropout = trial.suggest_float('dropout', 0.0, 0.5)
            learning_rate = trial.suggest_float('learning_rate', 1e-4, 1e-2, log=True)
            pooling_method = trial.suggest_categorical('pooling_method', ['mean', 'max', 'add'])
            sage = trial.suggest_categorical('sage', [True, False])
            # Classifier hyperparameters - permanently set to True for better performance
            # To make this optimizable again, change [True] to [True, False]
            use_two_layer_classifier = trial.suggest_categorical('use_two_layer_classifier', [True])
            
            # Positional encoding hyperparameters (if enabled and optimizing)
            pos_config = {}
            if USE_POSITIONAL_ENCODING and args.optimize_pos_encoding:
                depth_embedding_dim = trial.suggest_categorical('depth_embedding_dim', [16, 32, 64])
                child_embedding_dim = trial.suggest_categorical('child_embedding_dim', [16, 32, 64])
                pos_config.update({
                    'depth_embedding_dim': depth_embedding_dim,
                    'child_embedding_dim': child_embedding_dim
                })
            
            # Create model configuration
            config = {
                'embedding_dim': embedding_dim,
                'hidden_dim_1': hidden_dim_1,
                'hidden_dim_2': hidden_dim_2,
                'dropout': dropout,
                'pooling_method': pooling_method,
                'sage': sage,
                'use_two_layer_classifier': use_two_layer_classifier,
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
                # Training
                model.train()
                train_loss = 0.0
                train_batch_count = 0
                
                for batch in train_dataloader:
                    batch = batch.to(DEVICE)
                    optimizer.zero_grad()
                    
                    # Forward pass with optional positional encodings
                    if USE_POSITIONAL_ENCODING and has_pos_encodings and hasattr(batch, 'node_depth'):
                        output = model(batch.x, batch.edge_index, batch.batch, 
                                     batch.node_depth, batch.child_index)
                    else:
                        output = model(batch.x, batch.edge_index, batch.batch)
                    
                    loss = criterion(output.squeeze(), batch.y.float())
                    loss.backward()
                    optimizer.step()
                    
                    train_loss += loss.item()
                    train_batch_count += 1
                
                # Validation
                val_loss, _ = enhanced_evaluate(
                    model, val_dataloader, criterion, metrics, USE_POSITIONAL_ENCODING, has_pos_encodings
                )
                
                scheduler.step(val_loss)
                
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                
                # Report intermediate value for pruning
                trial.report(val_loss, epoch)
                
                # Handle pruning based on the intermediate value
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
        checkpoint_path = f"models/gcn_enhanced/{MODEL_NAME}/{MODEL_NAME}_best.pth"
        
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
        is_pos_model = isinstance(model, GCNWithPositionalEncoding)
        has_pos_data = hasattr(test_dataloader, 'dataset_obj') and hasattr(test_dataloader.dataset_obj, 'has_positional_encodings') and test_dataloader.dataset_obj.has_positional_encodings()
        
        print(f"Model uses positional encodings: {is_pos_model}")
        print(f"Dataset has positional encodings: {has_pos_data}")
        
        if is_pos_model and not has_pos_data:
            print("Warning: Model expects positional encodings but dataset doesn't provide them. Using zero embeddings.")
        
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
                writer.add_scalar(f"eval/test_{metric_name}", metric_value, epoch)
            writer.close()

        print("\n" + "=" * 50)
        print("ENHANCED GCN EVALUATION RESULTS:")
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
            print("Running hyperparameter optimization...")
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
                writer.add_scalar("optuna/best_value", study.best_value, study.best_trial.number)
                for key, value in study.best_trial.params.items():
                    writer.add_scalar(f"optuna/best_{key}", value, study.best_trial.number)

            pruned_trials = study.get_trials(deepcopy=False, states=[TrialState.PRUNED])
            complete_trials = study.get_trials(deepcopy=False, states=[TrialState.COMPLETE])

            print("Study statistics: ")
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
            if args.use_best_params:
                # Load best hyperparameters from Optuna study
                print("Loading best hyperparameters from Optuna study...")
                source_study_name = args.source_model_name if args.source_model_name else args.study_name
                
                try:
                    study = optuna.create_study(
                        storage=args.storage_url,
                        study_name=source_study_name,
                        direction="minimize",
                        load_if_exists=True
                    )
                    best_params = study.best_trial.params
                    print(f"Best parameters from study '{source_study_name}': {best_params}")
                    
                    # Create model configuration from best params
                    config = {
                        'embedding_dim': best_params.get('embedding_dim', 256),
                        'hidden_dim_1': best_params.get('hidden_dim_1', 128),
                        'hidden_dim_2': best_params.get('hidden_dim_2', 128),
                        'dropout': best_params.get('dropout', 0.1),
                        'pooling_method': best_params.get('pooling_method', 'mean'),
                        'sage': best_params.get('sage', False),
                        'use_two_layer_classifier': best_params.get('use_two_layer_classifier', False),
                        'learning_rate': best_params.get('learning_rate', 0.001)
                    }
                    
                    # Add positional encoding params if available
                    if USE_POSITIONAL_ENCODING:
                        config.update({
                            'depth_embedding_dim': best_params.get('depth_embedding_dim', 32),
                            'child_embedding_dim': best_params.get('child_embedding_dim', 32)
                        })
                    
                    model = create_enhanced_model_from_config(
                        num_node_features=train_loader.num_node_features,
                        config=config,
                        model_name=MODEL_NAME,
                        use_positional_encoding=USE_POSITIONAL_ENCODING
                    )
                    
                    optimizer = torch.optim.Adam(model.parameters(), lr=config['learning_rate'])
                    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5, verbose=True)
                    
                except Exception as e:
                    print(f"Failed to load best parameters: {e}")
                    print("Using dataset-based configuration...")
                    model, optimizer, scheduler = create_enhanced_model_with_dataset_config(
                        train_loader, MODEL_NAME, use_positional_encoding=USE_POSITIONAL_ENCODING
                    )
            else:
                # Create model with custom or default parameters
                config = {
                    'embedding_dim': args.embedding_dim,
                    'hidden_dim_1': args.hidden_dim_1,
                    'hidden_dim_2': args.hidden_dim_2,
                    'dropout': args.dropout,
                    'pooling_method': args.pooling_method,
                    'sage': args.sage,
                    'use_two_layer_classifier': args.two_layer_classifier,
                    'learning_rate': args.learning_rate
                }
                
                # Add positional encoding parameters
                if USE_POSITIONAL_ENCODING:
                    if args.max_depth is not None:
                        config['max_depth'] = args.max_depth
                    if args.max_child_index is not None:
                        config['max_child_index'] = args.max_child_index
                    config.update({
                        'depth_embedding_dim': args.depth_embed_dim,
                        'child_embedding_dim': args.child_embed_dim
                    })
                
                model, optimizer, scheduler = create_enhanced_model_with_dataset_config(
                    train_loader, MODEL_NAME, custom_config=config, use_positional_encoding=USE_POSITIONAL_ENCODING
                )
                
            criterion = torch.nn.BCEWithLogitsLoss()
            metrics = get_metrics()
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
                use_positional_encoding=USE_POSITIONAL_ENCODING,
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
                model, optimizer, save_path="models/gcn_enhanced", scheduler=scheduler
            )
            
            # Check dataset compatibility for evaluation
            is_pos_model = isinstance(model, GCNWithPositionalEncoding)
            has_pos_data = hasattr(test_dataloader, 'dataset_obj') and hasattr(test_dataloader.dataset_obj, 'has_positional_encodings') and test_dataloader.dataset_obj.has_positional_encodings()
            
            test_loss, test_metrics = enhanced_evaluate(
                model, test_dataloader, criterion, metrics, is_pos_model, has_pos_data
            )

            # Log final test results
            if writer is not None:
                writer.add_scalar("final/test_loss", test_loss, epoch)
                for metric_name, metric_value in test_metrics.items():
                    writer.add_scalar(f"final/test_{metric_name}", metric_value, epoch)

            print("\n" + "=" * 50)
            print("FINAL TEST RESULTS:")
            print("=" * 50)
            print(f"Test Loss: {test_loss:.5f}")
            for metric_name, metric_value in test_metrics.items():
                print(f"Test {metric_name}: {metric_value:.6f}")
            print("=" * 50)

        elif args.resume:
            print("Resuming training from checkpoint...")
            
            # Load model from checkpoint
            checkpoint_path = f"models/gcn_enhanced/{MODEL_NAME}/{MODEL_NAME}_best.pth"
            
            try:
                model, optimizer, scheduler, start_epoch, best_vloss, best_vacc = create_enhanced_model_from_checkpoint(
                    checkpoint_path, model_name=MODEL_NAME
                )
                print(f"Resumed from epoch {start_epoch}")
                print(f"Best validation loss: {best_vloss:.4f}, Best validation accuracy: {best_vacc:.4f}")
            except FileNotFoundError:
                print(f"Error: No saved model found for {MODEL_NAME}")
                print(f"Expected path: {checkpoint_path}")
                sys.exit(1)
            except Exception as e:
                print(f"Error loading model: {e}")
                sys.exit(1)
            
            # Continue training
            criterion = torch.nn.BCEWithLogitsLoss()
            metrics = get_metrics()
            
            # Check if model uses positional encodings
            use_pos_encoding = isinstance(model, GCNWithPositionalEncoding)
            
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
                start_epoch=start_epoch + 1,
                use_positional_encoding=use_pos_encoding,
            )

        # Close tensorboard writer
        if writer is not None:
            writer.close()