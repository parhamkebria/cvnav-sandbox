from types import SimpleNamespace

cfg = SimpleNamespace(
    data_root = "/home/parham/auairDataset/images",
    annotation_root = "/home/parham/auairDataset/annotations/annotation_files",
    seq_len = 4,          # number of input frames
    pred_steps = 1,       # predict 1 next frame
    img_size = (1920,1080),

    # model
    patch_size = 16,      # conv downsample factor
    latent_h = 48,        # latent grid height (approx) after encoder
    latent_w = 80,        # latent grid width (approx) after encoder
    vq_codebook_size = 512,
    d_model = 768,
    n_layers = 12,
    n_heads = 12,
    dropout = 0.1,

    # training
    batch_size = 2,  # Per-GPU batch size
    epochs = 3,
    lr = 1e-4,
    weight_decay = 1e-2,
    image_loss_weight = 1.0,
    nav_loss_weight = 1.0,
    device = "cuda" if __import__("torch").cuda.is_available() else "cpu",

    # misc
    save_every = 1,
    checkpoint_dir = "./checkpoints",
    num_workers = 8,  # Increased for multi-GPU
    
    # testing/debugging
    dataset_fraction = 0.05,  # Use 1.0 for full dataset, 0.1 for 10% subset for testing
)
