import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from config.graph_transformer_config import parse_args


if __name__ == "__main__":
    args = parse_args()
    MODEL_NAME = args.model_name
    SUFFIX = args.data_suffix
    DATASET = args.dataset
    import optuna
    from optuna.trial import TrialState
    import torch
    from models.GraphTransformer import GraphTransformer
    from data.dataset.graph_codet import GraphCoDeTM4
    from data.dataset.graph_aigcodeset import GraphAIGCodeSet
    from torch_geometric.loader import DataLoader
    from torchmetrics import Accuracy, Precision, Recall, Specificity, AUROC
    from utils.graph_transformer_utils import (
        save_model,
        load_model,
        get_metrics,
        set_seed,
        evaluate,
        train,
        load_single_data,
        load_multiple_data,
        create_objective,
        create_model_with_optuna_params,
        create_model_from_checkpoint,
        DEVICE,
    )
    from tqdm import tqdm
    import gc
    from torch.utils.tensorboard import SummaryWriter

    set_seed(args.seed)
    
    print(f"Using dataset: {DATASET}")
    print(f"Data directory: {args.data_dir}")
    
    # Initialize tensorboard writer if not disabled
    writer = None
    if not args.disable_tensorboard:
        log_dir = os.path.join(args.log_dir, MODEL_NAME)
        writer = SummaryWriter(log_dir=log_dir)
        print(f"Tensorboard logging enabled: {log_dir}")
    
    if args.eval:
        # Evaluation mode only - load model and evaluate on test set
        print("Running in evaluation mode...")
        
        # Load test data
        test_dataloader = load_single_data(
            data_dir=args.data_dir,
            split="test", 
            shuffle=False,
            batch_size=args.batch_size,
            suffix=SUFFIX,
            dataset=DATASET,
        )
        
        # Load model directly from checkpoint (includes architecture info)
        checkpoint_path = f"models/graph_transformer/{MODEL_NAME}/{MODEL_NAME}_best.pth"
        
        try:
            model, optimizer, scheduler, epoch, best_vloss, best_vacc = create_model_from_checkpoint(
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
        
        # Evaluate on test set
        criterion = torch.nn.BCEWithLogitsLoss()
        metrics = get_metrics()
        test_loss, test_metrics = evaluate(model, test_dataloader, criterion, metrics)

        # Log evaluation results to tensorboard
        if writer is not None:
            writer.add_scalar("eval/test_loss", test_loss, epoch)
            for metric_name, metric_value in test_metrics.items():
                writer.add_scalar(f"eval/test_{metric_name.lower()}", metric_value, epoch)
            writer.close()

        print("\n" + "=" * 50)
        print("EVALUATION RESULTS:")
        print("=" * 50)
        print(f"Test Loss: {test_loss:.5f}")
        for metric_name, metric_value in test_metrics.items():
            print(f"Test {metric_name}: {metric_value:.6f}")
        print("=" * 50)
        
    else:
        # Training or optimization mode
        train_loader, val_loader = load_multiple_data(
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
            objective = create_objective(
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
            print("Training model...")
            if args.use_best_params:
                # Create model with best hyperparameters from Optuna
                print(train_loader.num_node_features)
                model, optimizer, scheduler, optuna_success = create_model_with_optuna_params(
                    num_node_features=train_loader.num_node_features,
                    storage_url=args.storage_url,
                    study_name=args.study_name,
                    model_name=MODEL_NAME,
                    use_default_on_failure=False,  # Don't fallback to defaults in training mode
                    source_study_name=args.source_model_name  # Use source model name if provided
                )
                
                if not optuna_success:
                    source_info = f" from source model '{args.source_model_name}'" if args.source_model_name else ""
                    study_info = args.source_model_name if args.source_model_name else args.study_name
                    print(f"Failed to load Optuna parameters for training mode{source_info}")
                    print(f"Make sure the study '{study_info}' exists in {args.storage_url}")
                    sys.exit(1)
                    
                # Don't create a default scheduler if Optuna didn't use one
                # This ensures training matches the optimized configuration
                if scheduler is None:
                    print("Optuna study didn't use a scheduler, training without scheduler...")
            else:
                # Use default model architecture
                model = GraphTransformer(
                    num_node_features=train_loader.num_node_features,
                    embedding_dim=args.embedding_dim,
                    hidden_dim=args.hidden_dim,
                    num_heads=args.num_heads,
                    num_layers=args.num_layers,
                    dropout=args.dropout,
                    pooling_method=args.pooling_method
                ).to(DEVICE)

                optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
                model.name = MODEL_NAME
                
                # Create default scheduler
                scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                    optimizer=optimizer, patience=5
                )
                
            criterion = torch.nn.BCEWithLogitsLoss()
            metrics = get_metrics()
            train(
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                criterion=criterion,
                train_dataloader=train_loader,
                val_dataloader=val_loader,
                metrics=metrics,
                num_epochs=args.epochs,
                writer=writer,
            )

            # Clean up RAM to make room for the evaluation data
            del train_loader, val_loader
            gc.collect()

            # Final evaluation on test set
            test_dataloader = load_single_data(
                data_dir=args.data_dir,
                split="test", 
                shuffle=False,
                batch_size=args.batch_size,
                suffix=SUFFIX,
                dataset=DATASET,
            )

            epoch, best_vloss, best_vacc = load_model(
                model, optimizer, save_path="models/graph_transformer", scheduler=scheduler
            )
            test_loss, test_metrics = evaluate(model, test_dataloader, criterion, metrics)

            # Log final test results
            if writer is not None:
                writer.add_scalar("final/test_loss", test_loss, epoch)
                for metric_name, metric_value in test_metrics.items():
                    writer.add_scalar(f"final/test_{metric_name.lower()}", metric_value, epoch)
                writer.close()

            print("\n" + "=" * 50)
            print("FINAL TEST RESULTS:")
            print("=" * 50)
            print(f"Test Loss: {test_loss:.5f}")
            for metric_name, metric_value in test_metrics.items():
                print(f"Test {metric_name}: {metric_value:.6f}")
            print("=" * 50)

        elif args.resume:
            print("Resuming model training...")
            
            # Load model directly from checkpoint (includes architecture info)
            checkpoint_path = f"models/graph_transformer/{MODEL_NAME}/{MODEL_NAME}_best.pth"
            
            try:
                model, optimizer, scheduler, start_epoch, best_vloss, best_vacc = create_model_from_checkpoint(
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
            
            # Respect the original scheduler configuration from checkpoint
            if scheduler is None:
                print("No scheduler was used in original training, continuing without scheduler...")
                
            criterion = torch.nn.BCEWithLogitsLoss()
            metrics = get_metrics()
            
            # Continue training for additional epochs
            print(f"Continuing training for {args.epochs} more epochs...")
            train(
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
            )

            # Clean up RAM to make room for the evaluation data
            del train_loader, val_loader
            gc.collect()

            # Final evaluation on test set
            test_dataloader = load_single_data(
                data_dir=args.data_dir,
                split="test", 
                shuffle=False,
                batch_size=args.batch_size,
                suffix=SUFFIX,
                dataset=DATASET,
            )

            epoch, best_vloss, best_vacc = load_model(
                model, optimizer, save_path="models/graph_transformer", scheduler=scheduler
            )
            test_loss, test_metrics = evaluate(model, test_dataloader, criterion, metrics)

            # Log resumed training final test results
            if writer is not None:
                writer.add_scalar("resumed/test_loss", test_loss, epoch)
                for metric_name, metric_value in test_metrics.items():
                    writer.add_scalar(f"resumed/test_{metric_name.lower()}", metric_value, epoch)
                writer.close()

            print("\n" + "=" * 50)
            print("RESUMED TRAINING - FINAL TEST RESULTS:")
            print("=" * 50)
            print(f"Test Loss: {test_loss:.5f}")
            for metric_name, metric_value in test_metrics.items():
                print(f"Test {metric_name}: {metric_value:.6f}")
            print("=" * 50)
