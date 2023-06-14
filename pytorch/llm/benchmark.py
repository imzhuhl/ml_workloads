import sys
import time
import torch
from PIL import Image
from transformers import (BertTokenizerFast, BertModel, GPT2TokenizerFast, GPT2Model,
                          ViTImageProcessor, ViTModel, AutoImageProcessor, ResNetForImageClassification)
import torch_blade

DRYRUN = 'dryrun' in sys.argv
CAST = 'cast' in sys.argv

device = 'cuda' if torch.cuda.is_available() else 'cpu'
batch = 2048 if torch.cuda.is_available() else 256

@torch.no_grad()
def run_model(name, tokenizer, model, input, steps):
    if DRYRUN:
        return
    
    # model = torch.compile(model)

    model.to(device)

    def run():
        try:
            encoded_input = tokenizer(input, return_tensors='pt')
            encoded_input.to(device)
            model(**encoded_input)

            best_throughput = 0
            for _ in range(steps):
                t0 = time.time()
                # encoded_input = tokenizer(input, return_tensors='pt')
                # encoded_input.to(device)
                model(**encoded_input)
                t1 = time.time()
                cur_throughput = len(input) / (t1 - t0)
                best_throughput = cur_throughput if cur_throughput > best_throughput else best_throughput
                print(name + ' ' + str(cur_throughput))
            
            print(f"best throughput: {best_throughput}")
        except:
            print(name + ' 0')

    if CAST:
        with torch.autocast(device_type=device, dtype=torch.bfloat16):
            run()
    else:
        run()


# from torch.profiler import profile, ProfilerActivity
# with profile(activities=[ProfilerActivity.CPU], record_shapes=True) as prof:
#     # run_model('bert-base-uncased',
#     #         BertTokenizerFast.from_pretrained('bert-base-uncased'),
#     #         BertModel.from_pretrained('bert-base-uncased'),
#     #         ['Paris is the capital of [MASK].']*batch, 10)
#     run_model('gpt2',
#           GPT2TokenizerFast.from_pretrained('gpt2'),
#           GPT2Model.from_pretrained('gpt2'),
#           ['Once upon a time,']*batch, 10)
# print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=20, top_level_events_only=False))


# softmax
# gelu

# run_model('bert-base-uncased',
#             BertTokenizerFast.from_pretrained('bert-base-uncased'),
#             BertModel.from_pretrained('bert-base-uncased'),
#             ['Paris is a very beautiful city, Paris is the capital of [MASK].']*batch, 10)

# run_model('gpt2',
#           GPT2TokenizerFast.from_pretrained('gpt2'),
#           GPT2Model.from_pretrained('gpt2'),
#           ['The taste of swordfish, the cat and you want to know, the next line of the lyrics are']*batch, 10)

image = Image.open('./cats.jpg')

# run_model('dino-vitb16',
#           ViTImageProcessor.from_pretrained('facebook/dino-vitb16'),
#           ViTModel.from_pretrained('facebook/dino-vitb16'),
#           [image]*32, 5)

run_model('resnet-50',
          AutoImageProcessor.from_pretrained("microsoft/resnet-50"),
          ResNetForImageClassification.from_pretrained("microsoft/resnet-50"),
          [image]*16, 10)


"""
bert
avx2
-------------------------------------------  ------------  ------------  ------------  ------------  ------------  ------------
                                       Name    Self CPU %      Self CPU   CPU total %     CPU total  CPU time avg    # of Calls
-------------------------------------------  ------------  ------------  ------------  ------------  ------------  ------------
                               aten::linear         0.76%      26.328ms        93.64%        3.240s       2.579ms          1256
                                aten::addmm        54.03%        1.869s        56.36%        1.950s       2.671ms           730
                                aten::copy_        13.59%     470.099ms        13.60%     470.360ms     175.836us          2675
                             aten::uniform_        11.97%     413.940ms        11.97%     413.940ms       2.835ms           146
                                 aten::gelu         6.33%     219.025ms         6.33%     219.025ms       1.825ms           120
                               aten::matmul         0.18%       6.363ms         4.65%     160.838ms     443.080us           363
                              aten::normal_         3.52%     121.691ms         3.52%     121.691ms      40.564ms             3
                                   aten::to         0.39%      13.481ms         3.23%     111.717ms      64.427us          1734
                             aten::_to_copy         0.25%       8.742ms         3.17%     109.783ms      86.716us          1266
                                  aten::add         1.93%      66.710ms         3.12%     107.925ms     291.689us           370
                           aten::layer_norm         0.04%       1.409ms         2.14%      73.929ms     295.716us           250
                    aten::native_layer_norm         2.03%      70.091ms         2.10%      72.520ms     290.080us           250
                                aten::clone         0.12%       4.226ms         1.90%      65.784ms     137.050us           480
                              aten::reshape         0.04%       1.225ms         1.70%      58.735ms     115.167us           510
                              aten::softmax         0.12%       4.255ms         1.47%      50.948ms     424.567us           120
                             aten::_softmax         1.46%      50.512ms         1.46%      50.512ms     420.933us           120
                                  aten::bmm         1.43%      49.556ms         1.43%      49.556ms     206.483us           240
                            aten::embedding         0.01%     324.000us         0.60%      20.685ms     689.500us            30
                         aten::index_select         0.58%      20.075ms         0.59%      20.269ms     675.633us            30
                           aten::contiguous         0.02%     603.000us         0.33%      11.252ms      93.767us           120
-------------------------------------------  ------------  ------------  ------------  ------------  ------------  ------------
Self CPU time total: 3.460s

avx512
-------------------------------------------  ------------  ------------  ------------  ------------  ------------  ------------
                                       Name    Self CPU %      Self CPU   CPU total %     CPU total  CPU time avg    # of Calls
-------------------------------------------  ------------  ------------  ------------  ------------  ------------  ------------
                               aten::linear         0.59%      20.762ms        91.79%        3.235s       2.577ms          1255
                                aten::addmm        53.56%        1.887s        55.78%        1.966s       2.693ms           730
                                aten::copy_        13.71%     483.108ms        13.72%     483.440ms     180.725us          2675
                             aten::uniform_        11.80%     415.723ms        11.80%     415.723ms       2.847ms           146
                                 aten::gelu         6.05%     213.271ms         6.05%     213.271ms       1.777ms           120
                               aten::matmul         0.26%       9.245ms         4.58%     161.282ms     439.460us           367
                              aten::normal_         3.47%     122.357ms         3.47%     122.357ms      40.786ms             3
                                   aten::to         0.44%      15.490ms         3.15%     110.997ms      64.012us          1734
                                  aten::add         1.92%      67.570ms         3.13%     110.351ms     298.246us           370
                             aten::_to_copy         0.24%       8.549ms         3.09%     109.069ms      86.152us          1266
                              aten::softmax         0.04%       1.381ms         3.03%     106.940ms     891.167us           120
                             aten::_softmax         3.02%     106.448ms         3.02%     106.448ms     887.067us           120
                           aten::layer_norm         0.04%       1.515ms         2.10%      74.083ms     296.332us           250
                    aten::native_layer_norm         1.99%      70.227ms         2.06%      72.568ms     290.272us           250
                                aten::clone         0.10%       3.420ms         1.83%      64.485ms     134.344us           480
                              aten::reshape         0.06%       2.022ms         1.62%      57.246ms     112.247us           510
                                  aten::bmm         1.40%      49.196ms         1.40%      49.197ms     204.988us           240
                           aten::contiguous         0.03%       1.184ms         0.33%      11.557ms      96.308us           120
                                 aten::view         0.24%       8.604ms         0.24%       8.604ms       3.309us          2600
                                    aten::t         0.07%       2.532ms         0.19%       6.566ms       8.995us           730
-------------------------------------------  ------------  ------------  ------------  ------------  ------------  ------------
Self CPU time total: 3.524s


gpt
avx2
-------------------------------------------  ------------  ------------  ------------  ------------  ------------  ------------
                                       Name    Self CPU %      Self CPU   CPU total %     CPU total  CPU time avg    # of Calls
-------------------------------------------  ------------  ------------  ------------  ------------  ------------  ------------
                                aten::addmm        42.44%        1.216s        70.98%        2.035s       2.731ms           745
                              aten::normal_        23.96%     686.755ms        23.96%     686.755ms      11.077ms            62
                                 aten::tanh         9.77%     280.176ms         9.77%     280.176ms       2.335ms           120
                                  aten::add         4.50%     128.962ms         5.57%     159.766ms     261.911us           610
                                  aten::mul         5.10%     146.103ms         5.28%     151.215ms     308.602us           490
                                aten::copy_         4.65%     133.215ms         4.65%     133.215ms      54.462us          2446
                               aten::matmul         0.36%      10.404ms         3.41%      97.812ms     376.200us           260
                                  aten::pow         2.22%      63.711ms         2.23%      63.837ms     531.975us           120
                                   aten::to         0.27%       7.613ms         2.07%      59.219ms      28.608us          2070
                             aten::_to_copy         0.31%       8.886ms         1.99%      56.956ms      42.953us          1326
                                  aten::bmm         1.68%      48.267ms         1.68%      48.267ms     201.113us           240
                              aten::softmax         0.05%       1.383ms         1.39%      39.938ms     332.817us           120
                                aten::clone         0.10%       2.831ms         1.39%      39.931ms      83.190us           480
                             aten::_softmax         1.38%      39.542ms         1.38%      39.542ms     329.517us           120
                              aten::reshape         0.07%       1.943ms         1.29%      37.107ms      74.214us           500
                           aten::layer_norm         0.04%       1.211ms         1.18%      33.962ms     135.848us           250
                    aten::native_layer_norm         1.08%      30.844ms         1.14%      32.751ms     131.004us           250
                                aten::where         0.49%      13.921ms         0.49%      13.927ms     116.058us           120
                                 aten::view         0.25%       7.051ms         0.25%       7.051ms       3.276us          2152
                           aten::contiguous         0.02%     567.000us         0.22%       6.338ms      52.817us           120
-------------------------------------------  ------------  ------------  ------------  ------------  ------------  ------------
Self CPU time total: 2.866s

avx512
-------------------------------------------  ------------  ------------  ------------  ------------  ------------  ------------
                                       Name    Self CPU %      Self CPU   CPU total %     CPU total  CPU time avg    # of Calls
-------------------------------------------  ------------  ------------  ------------  ------------  ------------  ------------
                                aten::addmm        43.61%        1.216s        72.79%        2.030s       2.728ms           744
                              aten::normal_        24.64%     686.958ms        24.64%     686.958ms      11.080ms            62
                                 aten::tanh         7.76%     216.509ms         7.76%     216.509ms       1.804ms           120
                                  aten::add         4.79%     133.557ms         5.88%     163.945ms     268.762us           610
                                  aten::mul         5.39%     150.304ms         5.57%     155.451ms     317.247us           490
                                aten::copy_         4.66%     129.818ms         4.66%     129.818ms      53.074us          2446
                               aten::matmul         0.22%       6.062ms         3.27%      91.219ms     367.819us           248
                                   aten::to         0.22%       6.038ms         2.12%      59.197ms      28.598us          2070
                             aten::_to_copy         0.32%       8.786ms         2.03%      56.718ms      42.774us          1326
                                  aten::bmm         1.76%      49.026ms         1.76%      49.026ms     204.275us           240
                              aten::softmax         0.04%       1.150ms         1.62%      45.208ms     376.733us           120
                             aten::_softmax         1.61%      44.803ms         1.61%      44.803ms     373.358us           120
                                  aten::pow         1.58%      44.085ms         1.59%      44.208ms     368.400us           120
                                aten::clone         0.11%       2.950ms         1.33%      37.161ms      77.419us           480
                           aten::layer_norm         0.04%       1.218ms         1.28%      35.693ms     142.772us           250
                    aten::native_layer_norm         1.17%      32.502ms         1.24%      34.475ms     137.900us           250
                              aten::reshape         0.06%       1.734ms         1.22%      34.138ms      68.276us           500
                                aten::where         0.48%      13.325ms         0.48%      13.328ms     111.067us           120
                                 aten::view         0.25%       6.914ms         0.25%       6.914ms       3.213us          2152
                           aten::contiguous         0.02%     628.000us         0.23%       6.462ms      53.850us           120
-------------------------------------------  ------------  ------------  ------------  ------------  ------------  ------------
Self CPU time total: 2.789s


avx512 加速 tanh
avx512 拖慢 softmax

bert softmax : [bs, 12, 9, 9], (bs * 12 * 9) times, 9-dim for each softmax
gpt2 softmax : [bs, 12, 5, 5], (bs * 12 * 5) times, 5-dim for each softmax

bert tanh : 
"""
