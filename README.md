# repetition-counting
[Paper preprint](https://arxiv.org/submit/3795435/view)
![Alt Text](./video.gif)

## General usage
        python rep-count.py --data [DATA_dir] 

## Full command
        python rep-count.py --data [DATA_dir] -j 18 -d 3 -f 30 --wins 256 --noverlap 1

## Hyper parameters

- -j (default=18)

    Joint number of the skeleton

- -d (default=2)

    Dimension of the joint

- -f (default=30)

    Video frequency

- --wins (default=256)

    Sliding window frequency

- --noverlap (default=1)

    Sliding window steps
        
