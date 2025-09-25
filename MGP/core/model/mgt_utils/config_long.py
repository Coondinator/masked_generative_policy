import argparse

def vq_args_parser():
    parser = argparse.ArgumentParser(description='Optimal Transport AutoEncoder',
                                     add_help=True,
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ## dataloader
    parser.add_argument('--dataname', type=str, default='kit', help='dataset directory')
    parser.add_argument('--batch-size', default=256, type=int, help='batch size')
    parser.add_argument('--window-size', type=int, default=64, help='training motion length')

    ## optimization
    parser.add_argument('--total-iter', default=200000, type=int, help='number of total iterations to run')
    parser.add_argument('--warm-up-iter', default=1000, type=int, help='number of total iterations for warmup')
    parser.add_argument('--lr', default=1e-3, type=float, help='max learning rate')
    parser.add_argument('--lr-scheduler', default=[30000, 60000, 150000], nargs="+", type=int,
                        help="learning rate schedule (iterations)")
    parser.add_argument('--gamma', default=0.1, type=float, help="learning rate decay")
    parser.add_argument('--weight-decay', default=0.0, type=float, help='weight decay')
    parser.add_argument("--commit", type=float, default=0.02, help="hyper-parameter for the commitment loss")
    parser.add_argument('--recons-loss', type=str, default='l1_smooth', help='reconstruction loss')

    ## vqvae arch
    parser.add_argument("--code-dim", type=int, default=16, help="embedding dimension")
    parser.add_argument("--nb-code", type=int, default=1024, help="nb of embedding")
    parser.add_argument("--mu", type=float, default=0.99, help="exponential moving average to update the codebook")
    parser.add_argument("--down-t", type=int, default=2, help="downsampling rate")
    parser.add_argument("--stride-t", type=int, default=2, help="stride size")
    parser.add_argument("--width", type=int, default=16, help="width of the network")
    parser.add_argument("--depth", type=int, default=3, help="depth of the network")
    parser.add_argument("--dilation-growth-rate", type=int, default=3, help="dilation growth rate")
    parser.add_argument("--output-emb-width", type=int, default=16, help="output embedding width")
    parser.add_argument('--vq-act', type=str, default='relu', choices=['relu', 'silu', 'gelu'],
                        help='dataset directory')
    parser.add_argument('--vq-norm', type=str, default=None, help='dataset directory')

    ## quantizer
    parser.add_argument("--quantizer", type=str, default='ema_reset', choices=['ema', 'orig', 'ema_reset', 'reset'],
                        help="eps for optimal transport")
    parser.add_argument('--beta', type=float, default=1.0, help='commitment loss in standard VQ')

    ## resume
    parser.add_argument("--resume-pth", type=str, default=None, help='resume pth for VQ')
    parser.add_argument("--resume-gpt", type=str, default=None, help='resume pth for GPT')

    ## output directory
    parser.add_argument('--out-dir', type=str, default='./ckpt/', help='output directory')
    parser.add_argument('--results-dir', type=str, default='visual_results/', help='output directory')
    parser.add_argument('--visual-name', type=str, default='baseline', help='output directory')
    parser.add_argument('--exp-name', type=str, default='exp_debug',
                        help='name of the experiment, will create a file inside out-dir')
    ## other
    parser.add_argument('--print-iter', default=1000, type=int, help='print frequency')
    parser.add_argument('--eval-iter', default=10000, type=int, help='evaluation frequency')
    parser.add_argument('--save-iter', default=20000, type=int, help='print frequency')
    parser.add_argument('--visual-iter', default=5000, type=int, help='print frequency')
    parser.add_argument('--seed', default=123, type=int, help='seed for initializing training.')
    
    parser.add_argument('--nb-vis', default=20, type=int, help='nb of visualizations')
    args, unknown = parser.parse_known_args()
    return args


def trans_args_parser():
    parser = argparse.ArgumentParser(description='Optimal Transport AutoEncoder training ',
                                     add_help=True,
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('--mse-weight', default=1.0, type=float, help='weight for mse loss.')
    
    ## dataloader

    parser.add_argument('--dataname', type=str, default='t2m', help='dataset directory')
    parser.add_argument('--batch-size', default=32, type=int, help='batch size')
    parser.add_argument('--fps', default=[20], nargs="+", type=int, help='frames per second')
    parser.add_argument('--seq-len', type=int, default=64, help='training length')

    ## optimization
    parser.add_argument('--total-iter', default=150000, type=int, help='number of total iterations to run')
    parser.add_argument('--warm-up-iter', default=1000, type=int, help='number of total iterations for warmup')
    parser.add_argument('--lr', default=1e-4, type=float, help='max learning rate')
    parser.add_argument('--lr-scheduler', default=[30000, 60000, 120000], nargs="+", type=int,
                        help="learning rate schedule (iterations)")
    parser.add_argument('--gamma', default=0.1, type=float, help="learning rate decay")

    parser.add_argument('--weight-decay', default=1e-6, type=float, help='weight decay')
    parser.add_argument('--decay-option', default='all', type=str, choices=['all', 'noVQ'],
                        help='disable weight decay on codebook')
    parser.add_argument('--optimizer', default='adamw', type=str, choices=['adam', 'adamw'],
                        help='disable weight decay on codebook')

    ## vqvae arch
    parser.add_argument("--code-dim", type=int, default=16, help="embedding dimension")
    parser.add_argument("--nb-code", type=int, default=1024, help="nb of embedding")
    parser.add_argument("--mu", type=float, default=0.99, help="exponential moving average to update the codebook")
    parser.add_argument("--down-t", type=int, default=2, help="downsampling rate")
    parser.add_argument("--stride-t", type=int, default=2, help="stride size")
    parser.add_argument("--width", type=int, default=16, help="width of the network")
    parser.add_argument("--depth", type=int, default=3, help="depth of the network")
    parser.add_argument("--dilation-growth-rate", type=int, default=3, help="dilation growth rate")
    parser.add_argument("--output-emb-width", type=int, default=16, help="output embedding width")
    parser.add_argument('--vq-act', type=str, default='relu', choices=['relu', 'silu', 'gelu'],
                        help='dataset directory')

    ## gpt arch
    parser.add_argument("--block-size", type=int, default=54, help="seq len")
    parser.add_argument("--embed-dim-gpt", type=int, default=256, help="embedding dimension")
    parser.add_argument("--state-dim", type=int, default=9, help="latent dimension in the clip feature")
    parser.add_argument("--comb-state-dim", type=int, default=128, help="latent dimension in the clip feature")
    parser.add_argument("--pc-dim", type=int, default=64, help="pc hidden dimension after encoder")
    parser.add_argument("--cond-length", type=int, default=5, help="the sum length t of the state and pc")

    parser.add_argument("--num-layers", type=int, default=4, help="nb of transformer layers")
    parser.add_argument("--num-local-layer", type=int, default=2, help="nb of transformer local layers")
    parser.add_argument("--n-head-gpt", type=int, default=8, help="nb of heads")
    parser.add_argument("--ff-rate", type=int, default=4, help="feedforward size")
    parser.add_argument("--drop-out-rate", type=float, default=0.1, help="dropout ratio in the pos encoding")

    ## quantizer
    parser.add_argument("--quantizer", type=str, default='ema_reset', choices=['ema', 'orig', 'ema_reset', 'reset'],
                        help="eps for optimal transport")
    parser.add_argument('--quantbeta', type=float, default=1.0, help='dataset directory')

    ## resume
    parser.add_argument("--resume-vq", type=str, default='output/vq/10000_net_last.pth', help='resume vq pth')
    parser.add_argument("--resume-trans", type=str, default=None, help='resume gpt pth')

    ## output directory
    parser.add_argument('--out-dir', type=str, default='output/trans', help='output directory')
    parser.add_argument('--exp-name', type=str, default='exp_debug',
                        help='name of the experiment, will create a file inside out-dir')
    parser.add_argument('--vq-name', type=str, default='VQVAE',
                        help='name of the generated dataset .npy, will create a file inside out-dir')
    parser.add_argument('--vq-dir', type=str, default='output/vq', choices=['relu', 'silu', 'gelu'],
                        help='dataset directory')
    ## other
    parser.add_argument('--print-iter', default=200, type=int, help='print frequency')
    parser.add_argument('--save-iter', default=200, type=int, help='save frequency')
    parser.add_argument('--eval-rand-iter', default=1000, type=int, help='evaluation frequency')
    parser.add_argument('--eval-iter', default=150000, type=int, help='evaluation frequency')
    parser.add_argument('--seed', default=123, type=int, help='seed for initializing training. ')
    parser.add_argument("--if-maxtest", action='store_true', help="test in max")
    parser.add_argument('--pkeep', type=float, default=.5, help='keep rate for gpt training')
    parser.add_argument('--eval-env', default=2000, type=int, help='evaluation env frequency')

    ## generator
    parser.add_argument('--text', type=str, help='text')
    parser.add_argument('--length', type=int, help='length')
    args, unknown = parser.parse_known_args()
    return args
