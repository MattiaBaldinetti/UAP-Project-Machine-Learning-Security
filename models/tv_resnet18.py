import torch.nn as nn
from torchvision.models import resnet18

def TV_ResNet18_CIFAR10(num_classes=10) -> nn.Module:
    # Crea l’architettura di ResNet-18 di torchvision pensata per ImageNet, ma senza i pesi allenati su ImageNet
    m = resnet18(weights=None)  # resnet18 è la funzione di torchvision che costruisce un nn.Module di tipo ResNet-18
                                # non caricare pesi pre-addestrati, inizializzare i pesi in modo casuale 
                                # (secondo l’inizializzazione standard di torchvision)
    """ 
    In questo momento m è una ResNet-18 standard ImageNet, quindi tipicamente contiene:
    conv1: 7×7, stride 2, padding 3 (adatta a input 224×224)
    bn1
    relu
    maxpool: MaxPool2d stride 2 (riduce subito la risoluzione)
    layer1..layer4: blocchi residui [2,2,2,2]
    avgpool: AdaptiveAvgPool2d((1,1))
    fc: Linear(512 → 1000) perché ImageNet ha 1000 classi
    Quindi, Output Shape [B,3,224,224] → ... → [B,1000]
    """ 


    # Sostituisce il primo layer (pensato per ImageNet) con uno adatto a 32×32
    m.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False) # Stai rimpiazzando l’attributo conv1 del modello.
            # Il vecchio layer (7×7 stride 2) viene “scartato” (non più referenziato) e non verrà più usato.
            # Il nuovo layer è:
                # input channels = 3 (RGB)
                # output channels = 64
                # kernel = 3×3
                # stride = 1 (non riduce dimensione)
                # padding = 1 (mantiene dimensione)
                # bias=False (perché subito dopo c’è BatchNorm, spesso si elimina il bias)
            # Su CIFAR-10 le immagini sono 32×32:
                # conv 7×7 stride 2 è troppo “aggressiva” (perde informazione subito)
                # conv 3×3 stride 1 è lo standard CIFAR-style (mantiene dettaglio)
    """ 
    Config del modello dopo questa riga. Ora m è “semi-adattata”:
    conv1: 3×3, stride 1, padding 1 ✅ (adatto a 32×32)
    bn1, relu: invariati ✅
    maxpool: ancora MaxPool2d ❌ (non ancora adattato)
    layer1..layer4: invariati ✅ 
    avgpool: invariato (AdaptiveAvgPool2d((1,1))) ✅
    fc: ancora out_features=1000 ❌
    """
                        
    # Elimina maxpool iniziale (su 32×32 ridurrebbe troppo presto la risoluzione)
    m.maxpool = nn.Identity() # Stai sostituendo maxpool con un layer che: restituisce l’input identico, non cambia nulla (è un “pass-through”)
            # In ResNet ImageNet: 
                # dopo conv1 (stride2) c’è maxpool (stride2)
                # riduzione totale iniziale molto forte
            # Su CIFAR-10:
                # conv1 è già stride1
                # se tenessi maxpool stride2, passeresti da 32×32 a 16×16 troppo presto
    """ 
    Config del modello dopo questa riga. Ora m ha uno “stem” CIFAR-friendly completo:
    conv1: 3×3, stride 1, padding 1 ✅ (adatto a 32×32)
    bn1, relu: invariati ✅
    maxpool: Identity() ✅ (nessun downsampling iniziale)
    layer1..layer4: invariati ✅
    avgpool: invariato (AdaptiveAvgPool2d((1,1))) ✅
    fc: ancora out_features=1000 ❌
    """

    # Cambia l’ultimo layer per avere 10 classi
    # m.fc.in_features in ResNet-18 è tipicamente 512
    # Stai rimpiazzando il layer finale:
        # da Linear(512 → 1000) (ImageNet)
        # a Linear(512 → num_classes) (CIFAR-10: 10)
    m.fc = nn.Linear(m.fc.in_features, num_classes)
    """ 
    Config del modello dopo questa riga. Ora m è completamente adattato a CIFAR-10::
    conv1: 3×3, stride 1, padding 1 ✅ (adatto a 32×32)
    bn1: invariato (BatchNorm2d(64)) ✅
    relu: invariati ReLU(inplace=True) ✅
    maxpool: Identity() ✅ (nessun downsampling iniziale)
    layer1..layer4: invariati ✅
    [ layer1: 2 BasicBlock (64) ✅
    layer2: 2 BasicBlock (128, stride2 nel primo + downsample) ✅
    layer3: 2 BasicBlock (256, stride2 nel primo + downsample) ✅
    layer4: 2 BasicBlock (512, stride2 nel primo + downsample) ✅ ] 
    avgpool: invariato (AdaptiveAvgPool2d((1,1))) ✅
    fc: Linear(512 → 10) ✅
    Quindi, Output Shape [B,3,32,32] → ... → [B,10]
    """

    # La funzione restituisce il modello costruito.
    # ES: model_tv = TV_ResNet18_CIFAR10() restituisce un nn.Module di ResNet-18 adattato a CIFAR-10
    return m