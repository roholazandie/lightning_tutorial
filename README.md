# Pytorch Lightning
## How to use?

For use gpu you should:
```
python vae.py --gpus 1
```
For more than one gpu:
```
python vae.py --gpus 4 --distributed_backend ddp
```

Using 16 bit precision (for larger batch sizes):
```
python vae.py --gpus 4 --distributed_backend ddp --precision 16
```
(for older versions of pytorch you need to install apex to make this work)


### The tutorial link
https://www.youtube.com/watch?v=QHww1JH7IDU