
/root/miniconda3/envs/torch/bin/python -c "import torch; print(torch.__version__)"
/root/miniconda3/envs/torch/bin/python gpt2.py --save torch

/root/miniconda3/envs/torch_opt/bin/python -c "import torch; print(torch.__version__)"
/root/miniconda3/envs/torch_opt/bin/python gpt2.py --save torch_opt

/root/miniconda3/envs/torch/bin/python gpt2.py --compare torch,torch_opt
