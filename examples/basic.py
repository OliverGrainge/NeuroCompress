from NeuroCompress import NeuroCompress
from NeuroCompress.compressors import DistillationCompress, PruningCompress, QATCompress
from torchvision.models import resnet18, ResNet18_Weights


model = resnet18(weights=ResNet18_Weights.DEFAULT)

# ====== Automated Compression ===================

compressor = NeuroCompress(model)
compressor.compress()
auto_compressed_model = compressor.best_model()


# ======= Use Individual Techniques ==============

compressor = DistillationCompress(model)
compressor.compress()
auto_compressed_model = compressor.best_model()

compressor = PruningCompress(model)
compressor.compress()
auto_compressed_model = compressor.best_model()

compressor = QATCompress(model)
compressor.compress()
auto_compressed_model = compressor.best_model()
