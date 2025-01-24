# train.py
import argparse
import os
from pathlib import Path
from typing import Dict,Optional
import pickle
import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR
from model import AdaptiveAnchorGAT, L2Regularizer
from utils import set_random_seed, print_metrics
from dataloader import train_mini_batch
from evaluation import batch_evaluation
from calculate import calculate_anchor_points


class TrainingConfig:
    """Configuration for A2GAT model training."""

    def __init__(self):
        self.parser = argparse.ArgumentParser(
            description='Training and Testing A2GAT Model on Bipartite Graphs'
        )
        self._add_arguments()

    def _add_arguments(self):
        """Add training arguments to parser."""
        # Dataset parameters
        self.parser.add_argument('--dataset', default='ML-100K', type=str,
                                 help='Dataset name for training')
        self.parser.add_argument('--data_root_dir', default='./dataset', type=str,
                                 help='Root directory of datasets')
        self.parser.add_argument('--model_root_dir', default='./models', type=str,
                                 help='Root directory for saving models')

        # Training parameters
        self.parser.add_argument('--lr', default=0.002, type=float,
                                 help='Learning rate')
        self.parser.add_argument('--patience', type=int, default=20,
                                 help='Early stopping patience epochs')
        self.parser.add_argument('--lr_decay_factor', type=float, default=0.9,
                                 help='Learning rate decay factor')
        self.parser.add_argument('--lr_decay_step_size', type=int, default=80,
                                 help='Learning rate decay step size')
        self.parser.add_argument('--max_epoch', default=200, type=int,
                                 help='Maximum training epochs')
        self.parser.add_argument('--batch_size', default=1000, type=int,
                                 help='Training batch size')
        self.parser.add_argument('--test_batch', default=4000, type=int,
                                 help='Testing batch size')

        # Model parameters
        self.parser.add_argument('--dim', default=64, type=int,
                                 help='Embedding dimension')
        self.parser.add_argument('--reg', default=0.005, type=float,
                                 help='L2 regularization coefficient')

        # Training control
        self.parser.add_argument('--batch_per_epoch', default=10000, type=int,
                                 help='Maximum training batches per epoch')
        self.parser.add_argument('--full_batch', default=1, type=int,
                                 help='Whether to train all edges in an epoch')
        self.parser.add_argument('--test_step', default=1, type=int,
                                 help='Evaluation frequency in epochs')

        # Evaluation parameters
        self.parser.add_argument('--topk', default='10', type=str,
                                 help='Top-K recommendation thresholds')
        self.parser.add_argument('--print_max_K', default=1, type=int,
                                 help='Print metrics only for maximum K')

        # System parameters
        self.parser.add_argument('--gpu', default='0', type=str,
                                 help='GPU device number')
        self.parser.add_argument('--seed', default=2025, type=int,
                                 help='Random seed')

        # Model saving
        self.parser.add_argument('--save_model', default=False, action='store_true',
                                 help='Whether to save model checkpoints')
        self.parser.add_argument('--overwrite_model', default=False, action='store_true',
                                 help='Whether to overwrite existing model')
        self.parser.add_argument('--do_test', default=1, type=int,
                                 help='Whether to perform evaluation')

    def parse_args(self) -> argparse.Namespace:
        """Parse and process command line arguments."""
        args = self.parser.parse_args()

        # Process paths
        args.input_dir = Path(args.data_root_dir) / args.dataset
        if args.save_model:
            args.model_dir = Path(args.model_root_dir) / args.dataset
            args.model_dir.mkdir(parents=True, exist_ok=True)

        # Process evaluation parameters
        args.topk = [int(k) for k in args.topk.split(',')]
        args.max_K = max(args.topk)
        args.target_metric = f'ndcg@{args.max_K}'
        #precision, recall, ndcg

        # Process batch settings
        if args.full_batch:
            args.batch_per_epoch = None

        return args


class A2GATTrainer:
    """Trainer class for A2GAT model."""

    def __init__(self, args: argparse.Namespace, device: str):
        self.args = args
        self.device = device
        # self.best_ndcg = 0.0
        self.best = 0.0
        self.best_metrics = {}

        # Load dataset
        self.load_dataset()

        # Initialize model and training components
        self.initialize_model()
        self.initialize_training_components()

    def load_dataset(self):
        """Load and prepare training and testing datasets."""
        print('Loading dataset...')


        # Load training data
        with open(self.args.input_dir / 'train.csr.pickle', 'rb') as f:
            self.csr_train = pickle.load(f)
            print(f'Train: {self.csr_train.shape}, {self.csr_train.nnz}')

        # Load testing data
        with open(self.args.input_dir / 'test.csr.pickle', 'rb') as f:
            self.csr_test = pickle.load(f)
            print(f'Test: {self.csr_test.shape}, {self.csr_test.nnz}')


    def initialize_model(self):
        """Initialize A2GAT model and move to device."""
        print('Initializing model...')

        # Calculate number of anchor points
        n_anchor = calculate_anchor_points(self.csr_train)

        # Prepare adjacency matrix
        P = torch.tensor(
            self.csr_train.todense(),
            dtype=torch.float32,
            device=self.device
        )

        # Initialize model
        self.model = AdaptiveAnchorGAT(
            embedding_dim=self.args.dim,
            num_users=self.csr_train.shape[0],
            num_items=self.csr_train.shape[1],
            num_anchors=n_anchor,
            anchor_dim=n_anchor // 2,
            adjacency_matrix=P
        ).to(self.device)

    def initialize_training_components(self):
        """Initialize optimizer, scheduler, loss and regularizer."""
        self.optimizer = Adam(self.model.parameters(), lr=self.args.lr)
        self.scheduler = StepLR(
            self.optimizer,
            step_size=self.args.lr_decay_step_size,
            gamma=self.args.lr_decay_factor
        )
        self.criterion = nn.CrossEntropyLoss()
        self.regularizer = L2Regularizer(self.args.reg)

    def train_epoch(self, epoch: int, edges: torch.Tensor) -> None:
        """Train model for one epoch.

        Args:
            epoch: Current epoch number
            edges: Edge indices for training
        """
        self.model.train()
        batch_iterator = train_mini_batch(
            edges,
            self.args.batch_size,
            self.args.batch_per_epoch
        )

        for batch_id, batch in enumerate(batch_iterator):
            if not self.args.full_batch and batch_id >= self.args.batch_per_epoch:
                break

            # Prepare batch data
            user_indices = batch[:, 0].to(self.device)
            pos_item_indices = batch[:, 1].to(self.device)

            # Forward pass
            predictions, factors = self.model(user_indices, pos_item_indices)

            # Calculate losses
            fit_loss = self.criterion(predictions, pos_item_indices)
            reg_loss = self.regularizer(factors)
            total_loss = fit_loss + reg_loss

            # Backward pass and optimization
            self.optimizer.zero_grad()
            total_loss.backward()
            self.optimizer.step()

            if batch_id % 100 == 0:
                print(f'Epoch {epoch}, Batch {batch_id}, Loss: {total_loss.item():.4f}')

    def evaluate(self, epoch: int) -> Dict[str, float]:
        """Evaluate model performance.

        Args:
            epoch: Current epoch number

        Returns:
            Dictionary containing evaluation metrics
        """
        print('-' * 20)
        print('Evaluating...')


        metrics = batch_evaluation(
            self.args,
            self.model,
            self.csr_test,
            self.csr_train,
            epoch,
            self.device
        )


        return metrics

    def save_model(self, epoch: int, model_path: Optional[Path] = None) -> None:
        """Save model checkpoint.

        Args:
            epoch: Current epoch number
            model_path: Path to save model (optional)
        """
        if not model_path:
            model_path = self.args.model_dir / (
                'model.pt' if self.args.overwrite_model
                else f'model.epoch.{epoch}.pt'
            )

        torch.save(self.model.state_dict(), model_path)
        print(f'Model saved to {model_path}')

    def save_embeddings(self) -> None:
        """Save learned embeddings to files."""
        embeddings_dir = self.args.input_dir / 'embeddings'
        embeddings_dir.mkdir(exist_ok=True)

        # Save user embeddings in batches
        batch_size = 4000
        with open(embeddings_dir / 'user_embeddings.dat', 'w') as f:
            for i in range(0, self.model.num_users, batch_size):
                user_ids = torch.arange(
                    i,
                    min(i + batch_size, self.model.num_users),
                    device=self.device
                )
                user_embeddings = self.model.get_user_embedding(user_ids).cpu()

                for user_id, emb in zip(user_ids.cpu().numpy(), user_embeddings):
                    line = f'u{user_id} {" ".join(map(str, emb.detach().numpy()))}\n'
                    f.write(line)

        # Save item embeddings
        item_embeddings = self.model.get_item_embedding().cpu()
        with open(embeddings_dir / 'item_embeddings.dat', 'w') as f:
            for item_id, emb in enumerate(item_embeddings):
                line = f'v{item_id} {" ".join(map(str, emb.detach().numpy()))}\n'
                f.write(line)

        print(f'Embeddings saved to {embeddings_dir}')

    def train(self) -> None:
        """Main training loop."""
        print('Starting training...')


        # Prepare edge indices for training
        edges = torch.tensor(
            np.stack(self.csr_train.nonzero(), axis=-1),
            dtype=torch.long
        )

        epochs_no_improve = 0
        for epoch in range(self.args.max_epoch):
            # Training phase

            self.train_epoch(epoch, edges)


            # Learning rate adjustment
            self.scheduler.step()

            # Evaluation phase
            if self.args.do_test and epoch % self.args.test_step == 0:
                metrics = self.evaluate(epoch)

                # Check for improvement
                if metrics[self.args.target_metric] >= self.best:
                    self.best_metrics = metrics.copy()
                    self.best = metrics[self.args.target_metric]
                    epochs_no_improve = 0

                    # Save model if required
                    if self.args.save_model:
                        self.save_model(epoch)
                else:
                    epochs_no_improve += 1
                    if epochs_no_improve >= self.args.patience:
                        print(f'Early stopping after {self.args.patience} epochs without improvement.')
                        break

                # Print metrics
                print(f'** Epoch {epoch} **')
                print_metrics(self.args, metrics, self.args.print_max_K)
                print(f'** Best performance: epoch {self.best_metrics["epoch"]} **')
                print_metrics(self.args, self.best_metrics, self.args.print_max_K)

        # Save final embeddings
        # self.save_embeddings()



def main():
    """Main function to run A2GAT training."""
    # Initialize configuration
    config = TrainingConfig()
    args = config.parse_args()
    print(args)

    # Set device
    device = 'cpu' if args.gpu == '-1' else 'cuda:0'
    if device != 'cpu':
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    # Set random seed
    set_random_seed(args.seed, device != 'cpu')

    # Initialize trainer and start training
    trainer = A2GATTrainer(args, device)
    trainer.train()


if __name__ == '__main__':
    main()