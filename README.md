# Branchy-TEE
## A secure  and efficient inference framework for deep learning in  untrustworthy cloud platforms, termed Branchy-TEE, which  aims to protect the confidentiality and integrity of data and  models of multiple participating actors throughout the inference  process using the Trusted Execution Environment (TEE).

Prerequisites:

  Gramine with SGX support requires several features from your system:
 ```
      the FSGSBASE feature of recent processors must be enabled in the Linux kernel;
      
      the Intel SGX driver must be built in the Linux kernel;
      
      Intel SGX SDK/PSW and (optionally) Intel DCAP must be installed.
      
      install gramine following https://gramine.readthedocs.io/en/latest/quickstart.html
  ```
  
  （1）	File description：
  ```
        Catalog: Multi-exits with Distillation
        
            train-distillation-exit-model_name.py： Training models with multiple exit mechanisms and improving model accuracy by distillation. 
            
            eval-exits-model_name.py: Model accuracy verification
            
            model_name including: Alexnet/MobileNet v1/v2/vgg-19/vgg-16/resnet-50/101/152

         Catalog: SecEDIF-model_name-native:
         
            predict.py: Performing Inference in SGX
            
            mprofile_timestamp.dat: ENCLAVE Peak Memory Statistics
            
            result.txt: inference result
```


  (2) Parameters and execution
  
  Multi-exits with Distillation
 ```

    parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
    parser.add_argument('--lr', default=0.0001, type=float, help='learning rate')
    parser.add_argument('--cuda', action='store_true', default=True, help =' use GPU?')
    parser.add_argument('--batch-size', default=100, type=int, help = "Batch Size for Training")
    parser.add_argument('--num-workers', default=2, type=int, help = 'num-workers')
    parser.add_argument('--net', type = str, choices=['LeNet5', 'AlexNet', 'VGG16','VGG19','ResNet18','ResNet34',   
                                                       'DenseNet','MobileNetv1','MobileNetv2'], default='MobileNetv1', help='net type')
    parser.add_argument('--epochs', type = int, default=100, help = 'Epochs')
    parser.add_argument('--resume', '-r', action='store_true', help='resume from checkpoint')
    parser.add_argument('--patience', '-p', type = int, default=7, help='patience for Early stop')
    parser.add_argument('--optim','-o',type = str, choices = ['sgd','adam','adamw'], default = 'adamw', help = 'choose optimizer')
 ```
 SecEDIF-model_name-native
  ```
    sgx.allowed_files = ["file:result.txt"] ：The PyTorch manifest template also contains sgx.allowed_files list. It specifies a single file unconditionally allowed by the enclave:
    
    Let’s prepare all the files needed to run PyTorch in an SGX enclave: make SGX=1
    
    The above command performs the following tasks:

    1. Generates the final SGX manifest file pytorch.manifest.sgx.
    2. Signs the manifest and generates the SGX signature file containing SIGSTRUCT (pytorch.sig).
    3. Creates a dummy EINITTOKEN token file pytorch.token (this file is used for backwards compatibility with SGX platforms with EPID and without Flexible Launch Control).
    
    After running the above commands and building all the required files, we can use gramine-sgx to launch the PyTorch workload inside an SGX enclave:

    gramine-sgx ./pytorch predict.py

    It will run exactly the same Python script but inside the SGX enclave. Again, you can verify that PyTorch ran correctly by examining result.txt.


 ```
 
 
