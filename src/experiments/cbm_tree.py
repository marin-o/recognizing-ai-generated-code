#!/usr/bin/env python3
"""
CBM StarCoder Tree Model Training Script
=========================================

This script trains and evaluates the CBM (CNN-BiLSTM) model with StarCoder/CodeBERT
backbone and tree-sitter AST features on the SemEval 2026 Task 13 dataset.

Features:
- Support for CodeBERT or StarCoder 3B backbones
- Tree-sitter AST feature integration (11 features)
- Hyperparameter optimization with Optuna
- Training, evaluation, and resumption modes
- TensorBoard logging and misclassification analysis

Usage:
    python cbm_tree.py --train --model-name my_model
    python cbm_tree.py --optimize --n-trials 100
    python cbm_tree.py --eval --model-name my_model
"""

import sys
import os
import gc
import logging

# Add src to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from config.cbm_tree_config import parse_args
import torch

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


if __name__ == "__main__":
    args = parse_args()
    MODEL_NAME = args.model_name
    
    # Import heavy dependencies only after arg parsing
    import optuna
    from optuna.trial import TrialState
    from optuna.samplers import TPESampler
    from torch.utils.tensorboard import SummaryWriter
    from torch.optim.lr_scheduler import StepLR, CosineAnnealingLR
    
    from models.cbm_starcoder_tree import CBMStarCoderTree, load_tokenizer
    from utils.cbm_tree_utils import (
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
        # Precomputed embeddings functions
        load_precomputed_dataset,
        create_precomputed_dataloaders,
        train_model_precomputed,
        evaluate_model_precomputed,
        create_precomputed_model_from_checkpoint,
    )
    from models.cbm_precomputed import CBMPrecomputed
    from tqdm import tqdm
    
    # Set seed and device
    set_seed(args.seed)
    device = get_device(args.device)
    
    # Initialize tensorboard writer if not disabled
    writer = None
    if not args.disable_tensorboard:
        log_dir = os.path.join(args.log_dir, "CBM_Tree", MODEL_NAME)
        writer = SummaryWriter(log_dir=log_dir)
        logger.info(f"Tensorboard logging enabled: {log_dir}")
    
    if args.eval:
        # Evaluation mode only - load model and evaluate on test set
        logger.info("Running in evaluation mode...")
        
        # Check if using precomputed embeddings
        use_precomputed = args.precomputed is not None
        
        if use_precomputed:
            logger.info(f"Using precomputed embeddings from: {args.precomputed}")
            # Load precomputed data
            _, _, test = load_precomputed_dataset(
                embeddings_path=args.precomputed,
                train_subset=args.train_subset,
                full_test_set=args.full_test_set,
                val_ratio=args.val_ratio,
                test_ratio=args.test_ratio
            )
            
            # Create dataloader for precomputed embeddings
            _, _, test_dataloader = create_precomputed_dataloaders(
                None, None, test, args.batch_size, args.num_workers
            )
            
            # Load precomputed model from checkpoint
            checkpoint_path = f"models/cbm_tree/{MODEL_NAME}/best_model.pth"
            
            try:
                model, optimizer, scheduler, epoch, best_vloss, best_vacc = create_precomputed_model_from_checkpoint(
                    checkpoint_path, device=device
                )
                logger.info(f"Loaded precomputed model from epoch {epoch}")
                logger.info(f"Best validation loss: {best_vloss:.4f}, Best validation accuracy: {best_vacc:.4f}")
            except FileNotFoundError:
                logger.error(f"No saved model found for {MODEL_NAME}")
                logger.error(f"Expected path: {checkpoint_path}")
                sys.exit(1)
            except ValueError as e:
                logger.error(f"Error: {e}")
                sys.exit(1)
            except Exception as e:
                logger.error(f"Error loading model: {e}")
                sys.exit(1)
        else:
            # Load data normally
            _, _, test = load_dataset(
                train_subset=args.train_subset,
                full_test_set=args.full_test_set,
                val_ratio=args.val_ratio,
                test_ratio=args.test_ratio,
                subtask=args.subtask
            )
            
            # Tokenize
            tokenizer = load_tokenizer(args.backbone_type)
            _, _, test = tokenize_datasets(None, None, test, tokenizer, args.max_length)
            
            # Create dataloader
            _, _, test_dataloader = create_dataloaders(None, None, test, args.batch_size, args.num_workers)
            
            # Load model directly from checkpoint
            checkpoint_path = f"models/cbm_tree/{MODEL_NAME}/best_model.pth"
            
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
        
        # Create analysis directory if analysis is enabled
        analysis_dir = os.path.join(args.analysis_dir, MODEL_NAME) if args.enable_misclassification_analysis else None
        
        # Perform evaluation with optional misclassification analysis
        if use_precomputed:
            test_results = evaluate_model_precomputed(
                model, test_dataloader, criterion, metrics, device,
                perform_analysis=args.enable_misclassification_analysis,
                analysis_dir=analysis_dir,
                model_name=MODEL_NAME
            )
        else:
            test_results = evaluate_model(
                model, test_dataloader, criterion, metrics, device,
                perform_analysis=args.enable_misclassification_analysis,
                analysis_dir=analysis_dir,
                model_name=MODEL_NAME
            )
        
        # Handle the case where analysis might be returned
        if isinstance(test_results, tuple):
            test_results, analysis_results = test_results
        else:
            analysis_results = None

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
        tokenizer = load_tokenizer(args.backbone_type)
        
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
                tree_feature_projection_dim = trial.suggest_categorical("tree_feature_projection_dim", [64, 128, 256])
                gradient_clip = trial.suggest_float("gradient_clip", 0.5, 2.0)
                scheduler_type = trial.suggest_categorical("scheduler", ["cosine", "step"])
                
                # Load and tokenize data for this trial
                trial_train, trial_val, _ = load_dataset(
                    train_subset=args.train_subset,
                    full_test_set=False,
                    val_ratio=args.val_ratio,
                    test_ratio=args.test_ratio,
                    subtask=args.subtask
                )
                
                trial_train, trial_val, _ = tokenize_datasets(
                    trial_train, trial_val, None, tokenizer, args.max_length
                )
                
                trial_train_loader, trial_val_loader, _ = create_dataloaders(
                    trial_train, trial_val, None, batch_size, args.num_workers
                )
                
                # Create model
                embedding_dim = 768 if args.backbone_type == "codebert" else 2048
                model = CBMStarCoderTree(
                    backbone_type=args.backbone_type,
                    embedding_dim=embedding_dim,
                    filter_sizes=filter_sizes,
                    lstm_hidden_dim=lstm_hidden_dim,
                    num_classes=2,
                    dropout_rate=dropout_rate,
                    freeze_backbone=args.freeze_backbone,
                    tree_feature_projection_dim=tree_feature_projection_dim
                ).to(device)
                
                optimizer = torch.optim.AdamW(
                    model.parameters(),
                    lr=learning_rate,
                    weight_decay=weight_decay,
                    eps=1e-8
                )
                
                if scheduler_type == "cosine":
                    scheduler = CosineAnnealingLR(optimizer, T_max=args.search_epochs)
                else:
                    scheduler = StepLR(optimizer, step_size=5, gamma=0.5)
                
                criterion = torch.nn.CrossEntropyLoss()
                metrics = get_metrics(device)
                
                # Training loop for optimization
                best_val_f1 = 0.0
                patience = 5
                patience_counter = 0
                
                epoch_progress = tqdm(range(args.search_epochs), desc=f"Trial {trial.number}", position=1, leave=False)
                
                for epoch in epoch_progress:
                    # Train
                    model.train()
                    for data in trial_train_loader:
                        inputs = {
                            'input_ids': data['input_ids'].to(device),
                            'attention_mask': data['attention_mask'].to(device)
                        }
                        labels = data['labels'].to(device)
                        codes = data['codes']
                        languages = data['languages']
                        
                        optimizer.zero_grad()
                        outputs = model(inputs, codes=codes, languages=languages)
                        loss = criterion(outputs, labels)
                        loss.backward()
                        
                        if gradient_clip > 0:
                            torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip)
                        
                        optimizer.step()
                    
                    # Validate
                    val_results = evaluate_model(model, trial_val_loader, criterion, metrics, device)
                    val_f1 = val_results['f1']
                    
                    # Update scheduler
                    if scheduler is not None:
                        scheduler.step()
                    
                    # Track best F1
                    if val_f1 > best_val_f1:
                        best_val_f1 = val_f1
                        patience_counter = 0
                    else:
                        patience_counter += 1
                    
                    # Report intermediate value
                    trial.report(val_f1, epoch)
                    
                    # Pruning
                    if trial.should_prune():
                        raise optuna.TrialPruned()
                    
                    # Early stopping
                    if patience_counter >= patience:
                        break
                
                # Clean up
                del model, optimizer, scheduler, trial_train_loader, trial_val_loader
                gc.collect()
                torch.cuda.empty_cache()
                
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
                for trial in study.trials:
                    writer.add_scalar("optuna/trial_value", trial.value if trial.value else 0, trial.number)

            # Save best hyperparameters
            save_hyperparameters(study.best_trial.params, study.best_trial.value, "models/cbm_tree", MODEL_NAME)

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
            
            # Check if using precomputed embeddings
            use_precomputed = args.precomputed is not None
            
            if use_precomputed:
                logger.info(f"Using precomputed embeddings from: {args.precomputed}")
                # Load precomputed data
                train, val, test = load_precomputed_dataset(
                    embeddings_path=args.precomputed,
                    train_subset=args.train_subset,
                    full_test_set=False,
                    val_ratio=args.val_ratio,
                    test_ratio=args.test_ratio
                )
                
                # Create dataloaders for precomputed embeddings
                train_dataloader, val_dataloader, test_dataloader = create_precomputed_dataloaders(
                    train, val, test, args.batch_size, args.num_workers
                )
            else:
                # Load data normally
                train, val, test = load_dataset(
                    train_subset=args.train_subset,
                    full_test_set=False,
                    val_ratio=args.val_ratio,
                    test_ratio=args.test_ratio,
                    subtask=args.subtask
                )
                
                # Tokenize
                train, val, test = tokenize_datasets(train, val, test, tokenizer, args.max_length)
                
                # Create dataloaders
                train_dataloader, val_dataloader, test_dataloader = create_dataloaders(
                    train, val, test, args.batch_size, args.num_workers
                )
            
            if args.use_best_params:
                # Load best parameters from Optuna study
                logger.info("Loading best hyperparameters from Optuna study...")
                if use_precomputed:
                    logger.error("--use-best-params is not supported with --precomputed mode yet")
                    sys.exit(1)
                model, optimizer, scheduler, best_params = create_model_with_optuna_params(
                    args.storage_url, args.study_name, MODEL_NAME,
                    backbone_type=args.backbone_type, device=device
                )
                gradient_clip = best_params.get('gradient_clip', args.gradient_clip)
            else:
                # Use default or command-line parameters
                logger.info("Using default/command-line hyperparameters...")
                if use_precomputed:
                    # Get embedding dimension from dataset
                    sample_embedding = train[0]['embedding']
                    embedding_dim = len(sample_embedding)
                    logger.info(f"Detected embedding dimension: {embedding_dim}")
                    
                    model = CBMPrecomputed(
                        embedding_dim=embedding_dim,
                        filter_sizes=args.filter_sizes,
                        lstm_hidden_dim=args.lstm_hidden_dim,
                        num_classes=2,
                        dropout_rate=args.dropout_rate,
                        tree_feature_projection_dim=args.tree_feature_projection_dim
                    ).to(device)
                else:
                    embedding_dim = 768 if args.backbone_type == "codebert" else 2048
                    model = CBMStarCoderTree(
                        backbone_type=args.backbone_type,
                        embedding_dim=embedding_dim,
                        filter_sizes=args.filter_sizes,
                        lstm_hidden_dim=args.lstm_hidden_dim,
                        num_classes=2,
                        dropout_rate=args.dropout_rate,
                        freeze_backbone=args.freeze_backbone,
                        tree_feature_projection_dim=args.tree_feature_projection_dim
                    ).to(device)
                
                optimizer = torch.optim.AdamW(
                    model.parameters(),
                    lr=args.learning_rate,
                    weight_decay=args.weight_decay,
                    eps=1e-8
                )
                
                scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs)
                gradient_clip = args.gradient_clip
                
            criterion = torch.nn.CrossEntropyLoss()
            metrics = get_metrics(device)
            
            # Train the model
            if use_precomputed:
                model = train_model_precomputed(
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
                    gradient_clip=gradient_clip,
                    log_interval=args.log_interval,
                    save_path="models/cbm_tree",
                    model_name=MODEL_NAME,
                    writer=writer
                )
            else:
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
                    gradient_clip=gradient_clip,
                    log_interval=args.log_interval,
                    save_path="models/cbm_tree",
                    model_name=MODEL_NAME,
                    writer=writer
                )

            # Clean up RAM
            del train, val
            gc.collect()

            # Final evaluation on test set
            if use_precomputed:
                test_results = evaluate_model_precomputed(model, test_dataloader, criterion, metrics, device)
            else:
                test_results = evaluate_model(model, test_dataloader, criterion, metrics, device)
            
            # Handle the case where analysis might be returned
            if isinstance(test_results, tuple):
                test_results, _ = test_results

            # Log final test results
            if writer is not None:
                writer.add_scalar("final/test_loss", test_results['loss'], args.epochs)
                for metric_name, metric_value in test_results.items():
                    if metric_name != 'loss':
                        writer.add_scalar(f"final/test_{metric_name}", metric_value, args.epochs)

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
                train_subset=args.train_subset,
                full_test_set=False,
                val_ratio=args.val_ratio,
                test_ratio=args.test_ratio,
                subtask=args.subtask
            )
            
            # Tokenize
            train, val, test = tokenize_datasets(train, val, test, tokenizer, args.max_length)
            
            # Create dataloaders
            train_dataloader, val_dataloader, test_dataloader = create_dataloaders(
                train, val, test, args.batch_size, args.num_workers
            )
            
            # Load model directly from checkpoint
            checkpoint_path = f"models/cbm_tree/{MODEL_NAME}/best_model.pth"
            
            try:
                model, optimizer, scheduler, start_epoch, best_vloss, best_vacc = create_model_from_checkpoint(
                    checkpoint_path, device=device
                )
                logger.info(f"Resuming from epoch {start_epoch}")
            except FileNotFoundError:
                logger.error(f"No saved model found for {MODEL_NAME}")
                logger.error(f"Expected path: {checkpoint_path}")
                sys.exit(1)
            except ValueError as e:
                logger.error(f"Error: {e}")
                sys.exit(1)
            except Exception as e:
                logger.error(f"Error loading model: {e}")
                sys.exit(1)
            
            if scheduler is None:
                scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs)
                
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
                save_path="models/cbm_tree",
                model_name=MODEL_NAME,
                writer=writer,
                initial_best_vloss=best_vloss,
                initial_best_vacc=best_vacc,
                start_epoch=start_epoch
            )

            # Clean up RAM
            del train, val
            gc.collect()

            # Final evaluation on test set
            test_results = evaluate_model(model, test_dataloader, criterion, metrics, device)
            
            # Handle the case where analysis might be returned
            if isinstance(test_results, tuple):
                test_results, _ = test_results

            # Log resumed training final test results
            if writer is not None:
                writer.add_scalar("resumed/test_loss", test_results['loss'], start_epoch + args.epochs)
                for metric_name, metric_value in test_results.items():
                    if metric_name != 'loss':
                        writer.add_scalar(f"resumed/test_{metric_name}", metric_value, start_epoch + args.epochs)

            logger.info("=" * 50)
            logger.info("RESUMED TRAINING - FINAL TEST RESULTS:")
            logger.info("=" * 50)
            logger.info(f"Test Loss: {test_results['loss']:.5f}")
            for metric_name, metric_value in test_results.items():
                if metric_name != 'loss':
                    logger.info(f"Test {metric_name.capitalize()}: {metric_value:.6f}")
            logger.info("=" * 50)
    
    # Close tensorboard writer
    if writer is not None:
        writer.close()
    
    logger.info("Script completed successfully!")
