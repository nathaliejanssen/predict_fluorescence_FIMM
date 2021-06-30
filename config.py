class config:
    '''Define hyperparameters to train the model
    
    Args:
        num_of_slices (int): The number of planes from the z-stack to use as an input
        cutoff_first_slices (int): Bottom planes from the z-stack to leave out
        cutoff_last_slices (int): Top planes from the z-stack to leave out
        
        img_size (int): Number of pixels used as image width/height
       
        epochs (int): Number of epochs to train the model
        bs (int): Batch size
        lr (int): Learning rate used with the Adam optimizer
        
        all_channels (bool): If False only a single channel is trained, if True all channels are trained at once
        channel (str): String defining which channel to train. nucleus, draq7, cellmask, or mitotracker
        
        loss (str): Either mse (mean squared error), mae (mean absolute error), or accuracy
        metrics (list): List of additional metrics to follow during training
        
        input_dir (str): Direction to cropped and converted images to use as input
        log_dir (str): Direction to save log files during training
        model_dir (str): Direction to save checkpoint models during training
        model_name(str): Name of final model in .h5 format
        history_filename (str): Name of the history file containing (loss) metrics during training in csv format
        
        train_spheroid, val_spheroid, test_spheroid (str): Randomized list of spheroids to use for training, validation, and testing
    '''
    # General
    num_of_slices = 1
    cutoff_first_slices = 3
    cutoff_last_slices = 10
    
    img_size = 1024
    
    # Training
    epochs = 50
    bs = 1
    lr = 1e-05       
    
    all_channels = True        
    channel = 'nucleus'        
    
    loss = 'mse'
    metrics = ['mae', 'accuracy']
    
    input_dir = '/data/Nathalie_preprocessed_png/'
    log_dir = '/data/njanssen/logs/allfluo'
    model_dir = '/data/njanssen/models/allfluo/model.{epoch:02d}--{val_loss:.2f}.h5'
    model_name = 'allfluo.h5'
    history_filename = 'allfluo.csv'
    
    train_spheroid = '|'.join(['r02c06', 'r02c07', 'r02c12', 'r02c15', 'r02c18', 'r02c19', 'r02c20', 'r02c22', 'r03c03', 'r03c08', 'r03c09', 'r03c10', 'r03c11', 'r03c12', 'r03c14', 'r03c15', 'r03c16', 'r03c19', 'r03c23', 'r04c04', 'r04c11', 'r04c12', 'r04c16', 'r05c08', 'r05c12', 'r05c13', 'r05c18', 'r05c20', 'r05c21', 'r05c23', 'r06c06', 'r06c07', 'r06c11', 'r06c15', 'r06c16', 'r06c17', 'r06c20', 'r07c09', 'r07c10', 'r07c11', 'r07c14', 'r07c15', 'r07c22', 'r08c09', 'r08c10', 'r08c13', 'r08c14', 'r08c15', 'r08c18', 'r08c19', 'r08c23', 'r09c07', 'r09c08', 'r09c09', 'r09c11', 'r09c12', 'r09c13', 'r09c15', 'r09c17', 'r09c21', 'r09c23', 'r10c08', 'r10c09', 'r10c10', 'r10c11', 'r10c16', 'r10c22', 'r11c08', 'r11c09', 'r11c12', 'r11c16', 'r11c18', 'r11c19', 'r12c09', 'r12c11', 'r12c12', 'r12c14', 'r12c17', 'r12c19', 'r13c06', 'r13c10', 'r13c12', 'r13c14', 'r13c17', 'r13c18', 'r13c19', 'r13c22', 'r14c14', 'r14c16', 'r14c21', 'r14c22', 'r15c12', 'r15c13', 'r15c16', 'r15c17', 'r15c20', 'r15c23'])
    val_spheroid = '|'.join(['r02c17', 'r03c13', 'r03c22', 'r04c08', 'r04c18', 'r04c20', 'r04c22', 'r05c10', 'r05c19', 'r06c14', 'r06c18', 'r06c19', 'r06c21', 'r06c22', 'r07c19', 'r07c21', 'r08c11', 'r08c22', 'r09c06', 'r09c16', 'r09c18', 'r10c12', 'r10c15', 'r11c10', 'r11c15', 'r11c17', 'r11c20', 'r12c13', 'r12c23', 'r14c07', 'r14c10', 'r14c13', 'r14c15', 'r14c17', 'r14c18', 'r14c19', 'r15c14', 'r15c22'])
    
    # Prediction
    test_spheroid = '|'.join(['r02c04', 'r02c10', 'r02c11', 'r02c13', 'r02c14', 'r02c21', 'r02c23', 'r03c04', 'r03c07', 'r04c10', 'r04c14', 'r04c15', 'r04c17', 'r04c21', 'r04c23', 'r05c09', 'r05c11', 'r05c14', 'r05c15', 'r05c16', 'r05c17', 'r06c09', 'r06c13', 'r07c07', 'r07c16', 'r07c17', 'r07c18', 'r08c08', 'r08c12', 'r08c16', 'r09c10', 'r09c19', 'r09c20', 'r09c22', 'r10c13', 'r10c18', 'r10c20', 'r10c21', 'r10c23', 'r11c11', 'r11c13', 'r12c07', 'r12c08', 'r12c16', 'r12c21', 'r13c07', 'r13c15', 'r13c20', 'r13c21', 'r13c23', 'r14c09', 'r14c12', 'r14c20', 'r15c07', 'r15c08', 'r15c10', 'r15c15', 'r16c03', 'r16c22'])
    path_to_trained_model = 'nucleus/nucleus_020621_1plane.h5'
    suffix_predicted_imgs = '_1plane_pred.png'
    path_to_predictions = 'nucleus/pred_020621/1-plane/'