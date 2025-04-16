import os
import torch
import torch.nn as nn
import torch.optim as optim
import unittest
from unittest.mock import MagicMock, patch
from optimized_learner import Learner, create_resnet_model

class MockDataset:
    def __init__(self, length=10):
        self.length = length
        self.classes = ['class1', 'class2']
    
    def __len__(self):
        return self.length

class TestLearner(unittest.TestCase):
    
    def setUp(self):
        # Create mock data loaders
        self.mock_train_loader = MagicMock()
        self.mock_train_loader.dataset = MockDataset()
        self.mock_train_loader.__iter__ = MagicMock(return_value=iter([
            (torch.randn(2, 3, 224, 224), torch.tensor([0, 1])) 
            for _ in range(5)
        ]))
        self.mock_train_loader.dataset.classes = ['class1', 'class2']
        
        self.mock_test_loader = MagicMock()
        self.mock_test_loader.dataset = MockDataset()
        self.mock_test_loader.__iter__ = MagicMock(return_value=iter([
            (torch.randn(2, 3, 224, 224), torch.tensor([0, 1])) 
            for _ in range(3)
        ]))
        
        # Create a simple model
        self.model = nn.Sequential(
            nn.Flatten(),
            nn.Linear(3*224*224, 2)
        )
        
        # Create optimizer, loss and scheduler
        self.optimizer = optim.SGD(self.model.parameters(), lr=0.01)
        self.criterion = nn.CrossEntropyLoss()
        self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=1)
        
        # Create temp directory for checkpoints
        os.makedirs("test_checkpoints", exist_ok=True)
        
        # Create learner
        self.learner = Learner(
            model=self.model,
            train_dataloader=self.mock_train_loader,
            test_dataloader=self.mock_test_loader,
            optimizer=self.optimizer,
            loss=self.criterion,
            scheduler=self.scheduler,
            work_dir="test_checkpoints",
            device=torch.device("cpu")
        )
    
    def tearDown(self):
        # Clean up temp files
        if os.path.exists("test_checkpoints/best_model.pth"):
            os.remove("test_checkpoints/best_model.pth")
        if os.path.exists("test_checkpoints"):
            os.rmdir("test_checkpoints")
    
    def test_initialization(self):
        """Test that the learner initializes correctly"""
        self.assertEqual(self.learner.class_names, ['class1', 'class2'])
        self.assertEqual(self.learner.device, torch.device("cpu"))
        self.assertEqual(self.learner.best_acc, 0)
        self.assertEqual(self.learner.counter, 0)
    
    def test_train(self):
        """Test that the train method executes without errors"""
        with patch.object(self.learner, '_save_model') as mock_save:
            history = self.learner.train(epochs=2)
            
            # Check that history contains expected keys
            self.assertIn('train_loss', history)
            self.assertIn('train_acc', history)
            self.assertIn('val_loss', history)
            self.assertIn('val_acc', history)
            
            # Check that we have the correct number of entries
            self.assertEqual(len(history['train_loss']), 2)
            
            # Verify that the model was saved
            mock_save.assert_called()
    
    def test_test(self):
        """Test that the test method executes without errors"""
        loss, accuracy = self.learner.test()
        self.assertIsInstance(loss, float)
        self.assertIsInstance(accuracy, float)
    
    @patch('PIL.Image.open')
    @patch('torchvision.transforms.Compose')
    def test_inference(self, mock_compose, mock_open):
        """Test that the inference method executes without errors"""
        # Mock the image processing pipeline
        mock_transform = MagicMock()
        mock_compose.return_value = mock_transform
        mock_transform.return_value = torch.randn(1, 3, 224, 224)
        
        # Mock the image opening
        mock_img = MagicMock()
        mock_open.return_value = mock_img
        
        # Make a prediction
        prediction = self.learner.inference("fake_image.jpg")
        
        # Check that we got a string result
        self.assertIsInstance(prediction, str)
        self.assertIn(prediction, ['class1', 'class2'])

if __name__ == '__main__':
    unittest.main()
