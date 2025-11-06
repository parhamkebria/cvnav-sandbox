from types import SimpleNamespace

cfg = SimpleNamespace(
    data_root = "/home/parham/auairDataset/images",
    annotation_root = "/home/parham/auairDataset/annotations/annotation_files",
    seq_len = 4,          # number of input frames
    pred_steps = 1,       # predict 1 next frame
    img_size = (224, 224), # VGG-16 standard input size

    # model
    patch_size = 16,      # conv downsample factor
    latent_h = 7,         # latent grid height for 224x224 input with VGG
    latent_w = 7,         # latent grid width for 224x224 input with VGG
    vq_codebook_size = 512,
    d_model = 768,
    n_layers = 12,
    n_heads = 12,
    dropout = 0.1,

    # training
    batch_size = 2,  # Reduce batch size for stability
    epochs = 3,
    lr = 5e-5,  # Reduced learning rate for stability
    weight_decay = 1e-3,  # Reduced weight decay
    image_loss_weight = 1.0,
    nav_loss_weight = 0.1,  # Reduce navigation loss weight
    vq_loss_weight = 0.25,  # Add explicit VQ loss weight
    max_grad_norm = 1.0,  # Add gradient clipping
    device = "cuda" if __import__("torch").cuda.is_available() else "cpu",

    # misc
    save_every = 1,
    checkpoint_dir = "./checkpoints",
    num_workers = 12,  # Increased for better data loading performance
    pin_memory = True,  # Enable for faster GPU transfer
    prefetch_factor = 4,  # Prefetch more batches
    
    # testing/debugging
    dataset_fraction = 0.1,  # Increased to 10% for better training
)
