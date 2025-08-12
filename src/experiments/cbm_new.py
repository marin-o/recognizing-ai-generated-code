import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from config.cbm_config import parse_args
import logging

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

if __name__ == "__main__":
    args = parse_args()
    MODEL_NAME = args.model_name
    
    import optuna
    from optuna.trial import TrialState
    from optuna.samplers import TPESampler
    import torch
    from transformers import RobertaTokenizer, RobertaModel
    from models.cbmclassifier import CBMClassifier
    from data.dataset.codet_m4 import CoDeTM4
    from torch.utils.data import DataLoader
    from torch.utils.tensorboard import SummaryWriter
    from torch.optim.lr_scheduler import StepLR, CosineAnnealingLR
    from utils.cbm_utils import (
        set_seed,
        get_device,
        get_metrics,
        save_model_checkpoint,
        load_model_checkpoint,
        save_hyperparameters,
        load_hyperparameters,
        train_model,
        evaluate_model,
        create_model_from_checkpoint,
        create_model_with_optuna_params,
        load_dataset,
        tokenize_datasets,
        create_dataloaders,
    )
    from tqdm import tqdm
    import gc

    # Set seed and device
    set_seed(args.seed)
    device = get_device(args.device)
    
    # Initialize tensorboard writer if not disabled
    writer = None
    if not args.disable_tensorboard:
        log_dir = os.path.join(args.log_dir, "CBM", MODEL_NAME)
        writer = SummaryWriter(log_dir=log_dir)
        logger.info(f"Tensorboard logging enabled: {log_dir}")
    
    if args.eval:
        # Evaluation mode only - load model and evaluate on test set
        logger.info("Running in evaluation mode...")
        
        # Load data
        _, _, test = load_dataset(
            cache_dir=args.cache_dir,
            train_subset=args.train_subset,
            full_test_set=args.full_test_set,
            val_ratio=args.val_ratio,
            test_ratio=args.test_ratio
        )
        
        # Tokenize
        tokenizer = RobertaTokenizer.from_pretrained(args.pretrained_model)
        _, _, test = tokenize_datasets(None, None, test, tokenizer, args.max_length)
        
        # Create dataloader
        _, _, test_dataloader = create_dataloaders(None, None, test, args.batch_size, args.num_workers)
        
        # Load model directly from checkpoint
        checkpoint_path = f"models/cbm/{MODEL_NAME}/best_model.pth"
        
        try:
            model, optimizer, scheduler, epoch, best_vloss, best_vacc = create_model_from_checkpoint(
                checkpoint_path, device=device
            )
            logger.info(f"Loaded model from epoch {epoch}")
            logger.info(f"Best validation loss: {best_vloss:.4f}, Best validation accuracy: {best_vacc:.4f}")
        except FileNotFoundError:
            logger.error(f"No saved model found for {MODEL_NAME}")
            logger.error(f"Expected path: {checkpoint_path}")
            sys.exit(1)
        except ValueError as e:
            logger.error(f"Error: {e}")
            logger.error("This checkpoint was saved with an older version that doesn't include architecture information.")
            sys.exit(1)
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            sys.exit(1)
        
        # Evaluate on test set
        criterion = torch.nn.CrossEntropyLoss()
        metrics = get_metrics(device)
        test_results = evaluate_model(model, test_dataloader, criterion, metrics, device)

        # Log evaluation results to tensorboard
        if writer is not None:
            writer.add_scalar("eval/test_loss", test_results['loss'], epoch)
            for metric_name, metric_value in test_results.items():
                if metric_name != 'loss':
                    writer.add_scalar(f"eval/test_{metric_name}", metric_value, epoch)
            writer.close()

        logger.info("=" * 50)
        logger.info("EVALUATION RESULTS:")
        logger.info("=" * 50)
        logger.info(f"Test Loss: {test_results['loss']:.5f}")
        for metric_name, metric_value in test_results.items():
            if metric_name != 'loss':
                logger.info(f"Test {metric_name.capitalize()}: {metric_value:.6f}")
        logger.info("=" * 50)
        
    else:
        # Training or optimization mode
        
        # Load tokenizer
        tokenizer = RobertaTokenizer.from_pretrained(args.pretrained_model)
        
        if args.optimize:
            logger.info("Running hyperparameter optimization...")
            
            def objective(trial):
                """Optuna objective function for hyperparameter optimization"""
                learning_rate = trial.suggest_float("learning_rate", 1e-5, 1e-2, log=True)
                batch_size = trial.suggest_categorical("batch_size", [8, 16, 32])
                weight_decay = trial.suggest_float("weight_decay", 1e-5, 1e-1, log=True)
                dropout_rate = trial.suggest_float("dropout_rate", 0.1, 0.7)
                lstm_hidden_dim = trial.suggest_categorical("lstm_hidden_dim", [128, 256, 512])
                filter_sizes = trial.suggest_categorical("filter_sizes", [256, 512, 768, 1024])
                gradient_clip = trial.suggest_float("gradient_clip", 0.5, 2.0)
                scheduler_type = trial.suggest_categorical("scheduler", ["cosine", "step"])
                
                # Load and tokenize data for this trial
                trial_train, trial_val, _ = load_dataset(
                    cache_dir=args.cache_dir,
                    train_subset=args.train_subset,
                    full_test_set=False,
                    val_ratio=args.val_ratio,
                    test_ratio=args.test_ratio
                )
                
                trial_train, trial_val, _ = tokenize_datasets(
                    trial_train, trial_val, None, tokenizer, args.max_length
                )
                
                trial_train_loader, trial_val_loader, _ = create_dataloaders(
                    trial_train, trial_val, None, batch_size, args.num_workers
                )
                
                # Create model
                base_model = RobertaModel.from_pretrained(args.pretrained_model)
                model = CBMClassifier(
                    base_model=base_model,
                    lstm_hidden_dim=lstm_hidden_dim,
                    filter_sizes=filter_sizes,
                    dropout_rate=dropout_rate
                ).to(device)
                
                optimizer = torch.optim.AdamW(
                    model.parameters(),
                    lr=learning_rate,
                    weight_decay=weight_decay,
                    eps=1e-8
                )
                
                if scheduler_type == "cosine":
                    scheduler = CosineAnnealingLR(optimizer, T_max=args.search_epochs, eta_min=1e-7)
                else:
                    scheduler = StepLR(optimizer, step_size=args.search_epochs//3, gamma=0.1)
                
                criterion = torch.nn.CrossEntropyLoss()
                metrics = get_metrics(device)
                
                # Training loop for optimization
                best_val_f1 = 0.0
                patience = 5
                patience_counter = 0
                
                epoch_progress = tqdm(range(args.search_epochs), desc=f"Trial {trial.number}", position=1, leave=False)
                
                for epoch in epoch_progress:
                    # Training
                    model.train()
                    total_loss = 0.0
                    
                    for data in trial_train_loader:
                        input_ids = data["input_ids"].to(device)
                        attention_mask = data["attention_mask"].to(device)
                        labels = data["target_binary"].to(device)
                        inputs = {"input_ids": input_ids, "attention_mask": attention_mask}
                        
                        optimizer.zero_grad()
                        outputs = model(inputs)
                        loss = criterion(outputs, labels)
                        loss.backward()
                        
                        torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip)
                        optimizer.step()
                        total_loss += loss.item()
                    
                    # Validation
                    val_results = evaluate_model(model, trial_val_loader, criterion, metrics, device)
                    scheduler.step()
                    
                    current_f1 = val_results['f1']
                    if current_f1 > best_val_f1:
                        best_val_f1 = current_f1
                        patience_counter = 0
                    else:
                        patience_counter += 1
                    
                    epoch_progress.set_postfix({
                        "val_f1": f"{current_f1:.4f}",
                        "best_f1": f"{best_val_f1:.4f}",
                        "patience": f"{patience_counter}/{patience}"
                    })
                    
                    if patience_counter >= patience:
                        epoch_progress.set_description(f"Trial {trial.number} (Early Stop)")
                        break
                    
                    trial.report(current_f1, epoch)
                    
                    if trial.should_prune():
                        epoch_progress.set_description(f"Trial {trial.number} (Pruned)")
                        epoch_progress.close()
                        raise optuna.exceptions.TrialPruned()
                
                epoch_progress.close()
                
                # Clean up
                del model, optimizer, scheduler, trial_train_loader, trial_val_loader
                gc.collect()
                
                return best_val_f1
            
            # Run optimization
            os.makedirs("optuna", exist_ok=True)
            study = optuna.create_study(
                storage=args.storage_url,
                study_name=args.study_name,
                direction="maximize",
                load_if_exists=True,
                sampler=TPESampler(seed=args.seed),
                pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=3)
            )
            
            study.optimize(objective, n_trials=args.n_trials)

            # Log optimization results
            if writer is not None:
                writer.add_scalar("optuna/best_value", study.best_trial.value, len(study.trials))
                for key, value in study.best_trial.params.items():
                    writer.add_scalar(f"optuna/best_params/{key}", value, len(study.trials))
                writer.close()

            # Save best hyperparameters
            save_hyperparameters(study.best_trial.params, study.best_trial.value, "models/cbm", MODEL_NAME)

            pruned_trials = study.get_trials(deepcopy=False, states=[TrialState.PRUNED])
            complete_trials = study.get_trials(deepcopy=False, states=[TrialState.COMPLETE])

            logger.info("Study statistics:")
            logger.info(f"  Number of finished trials: {len(study.trials)}")
            logger.info(f"  Number of pruned trials: {len(pruned_trials)}")
            logger.info(f"  Number of complete trials: {len(complete_trials)}")

            logger.info("Best trial:")
            logger.info(f"  Value: {study.best_trial.value:.4f}")
            logger.info("  Params:")
            for key, value in study.best_trial.params.items():
                logger.info(f"    {key}: {value}")

        elif args.train:
            logger.info("Training model...")
            
            # Load data
            train, val, test = load_dataset(
                cache_dir=args.cache_dir,
                train_subset=args.train_subset,
                full_test_set=False,
                val_ratio=args.val_ratio,
                test_ratio=args.test_ratio
            )
            
            # Tokenize
            train, val, test = tokenize_datasets(train, val, test, tokenizer, args.max_length)
            
            # Create dataloaders
            train_dataloader, val_dataloader, test_dataloader = create_dataloaders(
                train, val, test, args.batch_size, args.num_workers
            )
            
            if args.use_best_params:
                # Create model with best hyperparameters from Optuna
                model, optimizer, scheduler, optuna_success = create_model_with_optuna_params(
                    storage_url=args.storage_url,
                    study_name=args.study_name,
                    model_name=MODEL_NAME,
                    pretrained_model=args.pretrained_model,
                    device=device,
                    use_default_on_failure=False
                )
                
                if not optuna_success:
                    logger.error("Failed to load Optuna parameters for training mode")
                    logger.error(f"Make sure the study '{args.study_name}' exists in {args.storage_url}")
                    sys.exit(1)
                    
                if scheduler is None:
                    logger.info("Optuna study didn't use a scheduler, training without scheduler...")
            else:
                # Use default model architecture
                base_model = RobertaModel.from_pretrained(args.pretrained_model)
                model = CBMClassifier(
                    base_model=base_model,
                    lstm_hidden_dim=256,
                    filter_sizes=768,
                    dropout_rate=0.5
                ).to(device)

                optimizer = torch.optim.AdamW(
                    model.parameters(), 
                    lr=args.learning_rate,
                    weight_decay=args.weight_decay,
                    eps=1e-8
                )
                
                scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-7)
                
            criterion = torch.nn.CrossEntropyLoss()
            metrics = get_metrics(device)
            
            # Train the model
            model = train_model(
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                criterion=criterion,
                train_dataloader=train_dataloader,
                val_dataloader=val_dataloader,
                metrics=metrics,
                device=device,
                epochs=args.epochs,
                patience=args.patience,
                gradient_clip=args.gradient_clip,
                log_interval=args.log_interval,
                save_path="models/cbm",
                model_name=MODEL_NAME,
                writer=writer
            )

            # Clean up RAM
            del train_dataloader, val_dataloader, train, val
            gc.collect()

            # Final evaluation on test set
            test_results = evaluate_model(model, test_dataloader, criterion, metrics, device)

            # Log final test results
            if writer is not None:
                writer.add_scalar("final/test_loss", test_results['loss'], args.epochs)
                for metric_name, metric_value in test_results.items():
                    if metric_name != 'loss':
                        writer.add_scalar(f"final/test_{metric_name}", metric_value, args.epochs)
                writer.close()

            logger.info("=" * 50)
            logger.info("FINAL TEST RESULTS:")
            logger.info("=" * 50)
            logger.info(f"Test Loss: {test_results['loss']:.5f}")
            for metric_name, metric_value in test_results.items():
                if metric_name != 'loss':
                    logger.info(f"Test {metric_name.capitalize()}: {metric_value:.6f}")
            logger.info("=" * 50)

        elif args.resume:
            logger.info("Resuming model training...")
            
            # Load data
            train, val, test = load_dataset(
                cache_dir=args.cache_dir,
                train_subset=args.train_subset,
                full_test_set=False,
                val_ratio=args.val_ratio,
                test_ratio=args.test_ratio
            )
            
            # Tokenize
            train, val, test = tokenize_datasets(train, val, test, tokenizer, args.max_length)
            
            # Create dataloaders
            train_dataloader, val_dataloader, test_dataloader = create_dataloaders(
                train, val, test, args.batch_size, args.num_workers
            )
            
            # Load model directly from checkpoint
            checkpoint_path = f"models/cbm/{MODEL_NAME}/best_model.pth"
            
            try:
                model, optimizer, scheduler, start_epoch, best_vloss, best_vacc = create_model_from_checkpoint(
                    checkpoint_path, device=device
                )
                logger.info(f"Resuming from epoch {start_epoch}")
                logger.info(f"Best validation loss so far: {best_vloss:.4f}, Best validation accuracy: {best_vacc:.4f}")
            except FileNotFoundError:
                logger.error(f"No saved model found for {MODEL_NAME}")
                logger.error(f"Expected path: {checkpoint_path}")
                logger.error("Use --train mode to train a new model")
                sys.exit(1)
            except ValueError as e:
                logger.error(f"Error: {e}")
                logger.error("This checkpoint was saved with an older version.")
                sys.exit(1)
            except Exception as e:
                logger.error(f"Error loading model: {e}")
                sys.exit(1)
            
            if scheduler is None:
                logger.info("No scheduler was used in original training, continuing without scheduler...")
                
            criterion = torch.nn.CrossEntropyLoss()
            metrics = get_metrics(device)
            
            # Continue training for additional epochs
            logger.info(f"Continuing training for {args.epochs} more epochs...")
            model = train_model(
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                criterion=criterion,
                train_dataloader=train_dataloader,
                val_dataloader=val_dataloader,
                metrics=metrics,
                device=device,
                epochs=args.epochs,
                patience=args.patience,
                gradient_clip=args.gradient_clip,
                log_interval=args.log_interval,
                save_path="models/cbm",
                model_name=MODEL_NAME,
                writer=writer,
                initial_best_vloss=best_vloss,
                initial_best_vacc=best_vacc,
                start_epoch=start_epoch
            )

            # Clean up RAM
            del train_dataloader, val_dataloader, train, val
            gc.collect()

            # Final evaluation on test set
            test_results = evaluate_model(model, test_dataloader, criterion, metrics, device)

            # Log resumed training final test results
            if writer is not None:
                writer.add_scalar("resumed/test_loss", test_results['loss'], start_epoch + args.epochs)
                for metric_name, metric_value in test_results.items():
                    if metric_name != 'loss':
                        writer.add_scalar(f"resumed/test_{metric_name}", metric_value, start_epoch + args.epochs)
                writer.close()

            logger.info("=" * 50)
            logger.info("RESUMED TRAINING - FINAL TEST RESULTS:")
            logger.info("=" * 50)
            logger.info(f"Test Loss: {test_results['loss']:.5f}")
            for metric_name, metric_value in test_results.items():
                if metric_name != 'loss':
                    logger.info(f"Test {metric_name.capitalize()}: {metric_value:.6f}")
            logger.info("=" * 50)
