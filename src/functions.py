
"""
-------------------------------------------------------------------------------------------------
Supporting: Autoregressive Transformers for Data-Driven Spatio-Temporal Learning of Turbulent Flows
URL:        https://arxiv.org/abs/2209.08052
Author:     Aakash Patil, Jonathan Viquerat, Elie Hachem                             
Year:       March, 2023                                                
-------------------------------------------------------------------------------------------------
"""

import tensorflow as tf
import numpy as np
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input,Conv2D, Dense, Flatten, MaxPool2D, Activation, UpSampling2D

def tmodel_autoReg_run(model,input_shape,sampling_size, tend=2):
    inputs = Input(shape=input_shape)
    modelC = model(input_shape,sampling_size)
    predictions = []
    prediction = modelC(inputs)
    predictions.append(prediction)
 
    for t in range(1,tend):
        x = prediction
        prediction = modelC(x)
        predictions.append(prediction)

    predictions = tf.stack(predictions)
    print( "predictions.shape ", predictions.shape)
    predictions = tf.transpose(predictions, [1, 0, 2,3,4])
    print( "predictions.shape ", predictions.shape)
    model = Model(inputs=inputs, outputs=predictions)
    return model
#model_autoReg_run = tmodel_autoReg_run(model_convXformer,input_shape=(192,128,2),sampling_size=(24, 16),tend=5)
#print( model_autoReg_run.summary() )


def getTseq_inout(snapshots, tin=2, tnext=4):
    iparr,oparr=[],[] 
    for t in range(snapshots): 
        inar = [t+n for n in range(tin)] 
        outar = [ tin+t+n for n in range(tnext)]  
        iparr.append( inar ) 
        oparr.append( outar ) 
     
    delrows=[] 
    for i in range(len(iparr)): 
        tag = '' 
        if  (np.array(oparr[i])> snapshots-1).any()  : 
            tag = '<<<<' 
            delrows.append(i  ) 
        #print(iparr[i], oparr[i], tag) 
    iparr = np.delete(iparr, delrows, axis=0 ) 
    oparr = np.delete(oparr, delrows, axis=0 ) 
    #for i in range(len(iparr)): 
    #    print(iparr[i], oparr[i]) 
    print("Total ip,op sets: ", len(iparr), len(oparr))    
    return iparr, oparr;          


def make_2Dfrom3D(data, manyDz='all' ):
    #convert from 3D to 2D
    #stacking z cuts long time 
    assert  data.ndim == 5 #IS NOT A 3D dataset if assert fails
    ndZ = data.shape[3]
    ndT = data.shape[0]

    if manyDz=='all':
        lastDz = ndZ
        gapDz = 1
    else:
        lastDz = ndZ
        gapDz = lastDz//manyDz        

    print("lastDz, gapDz ",lastDz, gapDz)
    print("Cutting Data 3d to 2d along z (may take a while . . .)")
    znum = 0
    ds_inZ = data[...,znum,:] 
    for znum in tqdm(range(1,lastDz,gapDz )): # exclude 0 and 128 in Z dir
        ds_inZ = np.append(ds_inZ, data[...,znum,:],axis=0)  # exclude 0 and 192 in X dir

    data = np.array(ds_inZ)
    print("Print returning data with shape ",data.shape)
    return data;
#data = make_2Dfrom3D(data, manyDz='all')
#dataM1 = make_2Dfrom3D(data, manyDz=16)
#dataM1 = make_2Dfrom3D(data, manyDz=48)
  

def initWeights(output_size):
    b = np.zeros((2, 3), dtype='float32')
    b[0, 0] = 1
    b[1, 1] = 1
    W = np.zeros((output_size, 6), dtype='float32')
    weights = [W, b.flatten()]
    return weights;


def samplerblock(input_fmap, theta, out_dims=None, **kwargs):
    B = tf.shape(input_fmap)[0]
    H = tf.shape(input_fmap)[1]
    W = tf.shape(input_fmap)[2]
    theta = tf.reshape(theta, [B, 2, 3])
    if out_dims:
        out_H = out_dims[0]
        out_W = out_dims[1]
        batch_grids = bmmBlock(out_H, out_W, theta)
    else:
        batch_grids = bmmBlock(H, W, theta)
    x_s = batch_grids[:, 0, :, :]
    y_s = batch_grids[:, 1, :, :]

    out_fmap = samplingBilinear(input_fmap, x_s, y_s)

    return out_fmap;

def diff_loss_grad(y_pred, y_true):
  beta = 0.01
  dy_true, dx_true = tf.image.image_gradients(y_true)
  dy_pred, dx_pred = tf.image.image_gradients(y_pred)
  loss = tf.square(y_pred - y_true)
  loss = tf.math.reduce_mean(loss,axis=-1)+ tf.math.reduce_mean(tf.abs(dy_pred - dy_true) + tf.abs(dx_pred - dx_true), axis=-1)
  return loss

def bmmBlock(height, width, theta):
    num_batch = tf.shape(theta)[0]

    x = tf.linspace(-1.0, 1.0, width)
    y = tf.linspace(-1.0, 1.0, height)
    x_t, y_t = tf.meshgrid(x, y)

    x_t_flat = tf.reshape(x_t, [-1])
    y_t_flat = tf.reshape(y_t, [-1])

    ones = tf.ones_like(x_t_flat)
    sampling_grid = tf.stack([x_t_flat, y_t_flat, ones])

    sampling_grid = tf.expand_dims(sampling_grid, axis=0)
    sampling_grid = tf.tile(sampling_grid, tf.stack([num_batch, 1, 1]))

    theta = tf.cast(theta, 'float32')
    sampling_grid = tf.cast(sampling_grid, 'float32')

    batch_grids = tf.matmul(theta, sampling_grid)
    batch_grids = tf.reshape(batch_grids, [num_batch, 2, height, width])

    return batch_grids;

def join_3Dfrom2D_quick(data, nTimes):
    assert data.ndim == 4
    nT_by_nZ, NX, NY, NCH = data.shape
    assert     nT_by_nZ%nTimes == 0
    nZ = nT_by_nZ//nTimes
    data = data.reshape( nZ, nTimes, NX, NY, NCH)
 
    data = np.moveaxis(data, 0, 3)
    print("data.shape ", data.shape)
    return data;
#re3DdataM2 = join_3Dfrom2D_quick(dataM2, nTimes=196)

def get_cornerNode(data2d, x, y):
    shape = tf.shape(x)
    batch_size = shape[0]
    height = shape[1]
    width = shape[2]

    batch_idx = tf.range(0, batch_size)
    batch_idx = tf.reshape(batch_idx, (batch_size, 1, 1))
    b = tf.tile(batch_idx, (1, height, width))

    indices = tf.stack([b, y, x], 3)

    return tf.gather_nd(data2d, indices)

def doNorm_CHwise(XArr, minV=None, maxV=None):
    print("Inshape : ", XArr.shape)     
    print("**GLOBAL X.min( ) X.mean() X.max( )  ", XArr.min( ), XArr.mean(), XArr.max( ) )    
    NCH = XArr.shape[-1]
    XNEW = []
    for chn in range(NCH):
        print("For ch=",chn )
        THIS = doNorm(XArr[...,chn], minV, maxV)
        XNEW.append( THIS )
    XNEW = np.moveaxis(np.array(XNEW), 0,-1)
    print("Outshape : ", XNEW.shape)   
    print("**GLOBAL X.min( ) X.mean() X.max( )  ", XNEW.min( ), XNEW.mean(), XNEW.max( ) )      
    return XNEW;
#data2 = doNorm(data, minV=None, maxV=None)
#data2 = doNorm_CHwise(data, minV=None, maxV=None)


def samplingBilinear(data2d, x, y):

    H = tf.shape(data2d)[1]
    W = tf.shape(data2d)[2]
    max_y = tf.cast(H - 1, 'int32')
    max_x = tf.cast(W - 1, 'int32')
    zero = tf.zeros([], dtype='int32')

    x = tf.cast(x, 'float32')
    y = tf.cast(y, 'float32')
    x = 0.5 * ((x + 1.0) * tf.cast(max_x-1, 'float32'))
    y = 0.5 * ((y + 1.0) * tf.cast(max_y-1, 'float32'))

    x0 = tf.cast(tf.floor(x), 'int32')
    x1 = x0 + 1
    y0 = tf.cast(tf.floor(y), 'int32')
    y1 = y0 + 1

    x0 = tf.clip_by_value(x0, zero, max_x)
    x1 = tf.clip_by_value(x1, zero, max_x)
    y0 = tf.clip_by_value(y0, zero, max_y)
    y1 = tf.clip_by_value(y1, zero, max_y)

    Ia = get_cornerNode(data2d, x0, y0)
    Ib = get_cornerNode(data2d, x0, y1)
    Ic = get_cornerNode(data2d, x1, y0)
    Id = get_cornerNode(data2d, x1, y1)

    x0 = tf.cast(x0, 'float32')
    x1 = tf.cast(x1, 'float32')
    y0 = tf.cast(y0, 'float32')
    y1 = tf.cast(y1, 'float32')
    wa = (x1-x) * (y1-y)
    wb = (x1-x) * (y-y0)
    wc = (x-x0) * (y1-y)
    wd = (x-x0) * (y-y0)

    wa = tf.expand_dims(wa, axis=3)
    wb = tf.expand_dims(wb, axis=3)
    wc = tf.expand_dims(wc, axis=3)
    wd = tf.expand_dims(wd, axis=3)

    out = tf.add_n([wa*Ia, wb*Ib, wc*Ic, wd*Id])

    return out

def doNorm(X, minV=None, maxV=None):
    print(" X.min( ) X.mean() X.max( )  ", X.min( ), X.mean(), X.max( ) )
    X_std = (X - X.min()) / (X.max( ) - X.min( ))
    if maxV== None and minV== None:
        print("No maxV and minV given, returning based on [-1,1]")
        maxV = 1
        minV = -1
    X = X_std * (maxV - minV) + minV
    print(" X.min( ) X.mean() X.max( )  ", X.min( ), X.mean(), X.max( ) )
    return X;



def model_convXformer(input_shape=(192, 128, 2), sampling_size=(24, 16)):
    uCConv2D = Conv2D
    inputchannels = input_shape[-1]
    image = Input(shape=input_shape)
    locnet = uCConv2D(32, (5, 5), padding='same')(image)
    locnet = attention_block(locnet)

    locnet = Activation('relu')(locnet)
    locnet = MaxPool2D(pool_size=(2, 2))(locnet)
    locnet = uCConv2D(32, (5, 5), padding='same')(locnet)
    locnet = attention_block(locnet)

    locnet = Activation('relu')(locnet)
    locnet = MaxPool2D(pool_size=(2, 2))(locnet)
    locnet = uCConv2D(32, (5, 5), padding='same')(locnet)
    locnet = attention_block(locnet)

    locnet = Activation('relu')(locnet)
    locnet = MaxPool2D(pool_size=(2, 2))(locnet)
    locnet = Flatten()(locnet)
    locnet = Dense(500)(locnet)
    locnet = Activation('relu')(locnet)
    locnet = Dense(200)(locnet)
    locnet = Activation('relu')(locnet)
    locnet = Dense(100)(locnet)
    locnet = Activation('relu')(locnet)
    locnet = Dense(50)(locnet)
    locnet = Activation('relu')(locnet)
    weights = initWeights(50)
    locnet = Dense(6, weights=weights)(locnet)
    try:
        x = samplerblock(image,locnet, sampling_size) 
    except:
        x = samplerblock(image,locnet, sampling_size) 
    x = uCConv2D(32, (5, 5), padding='same')(x)
    x = attention_block(x)

    x = Activation('relu')(x)
    x = UpSampling2D (size=(2,2))(x)
    x = uCConv2D(32, (5, 5), padding='same')(x)
    x = attention_block(x)

    x = Activation('relu')(x)
    x = UpSampling2D (size=(2,2))(x)
    x = uCConv2D(32, (5, 5), padding='same')(x)
    x = attention_block(x)

    x = Activation('relu')(x)
    x = uCConv2D(32, (5, 5), padding='same')(x)
    x = attention_block(x)

    x = Activation('relu')(x)
    x = UpSampling2D (size=(2,2))(x)
    x = uCConv2D(inputchannels, (5, 5), padding='same')(x)

    x = Activation('linear')(x)
    return Model(inputs=image, outputs=x)
#testModel = model_convXformer(input_shape=(192, 128, 2), sampling_size=(24, 16))

def tmodel_autoReg(model,input_shape,sampling_size, tend=2):
    input = Input(shape=input_shape)
    output = model(input_shape,sampling_size)(input)
    for t in range(1,tend):
        output = model(input_shape,sampling_size)(output)
    model = Model(inputs=input, outputs=output)
    return model
#model_autoReg = tmodel_autoReg(model_convXformer,input_shape=(192,128,2),sampling_size=(24, 16),tend=5)
#print( model_autoReg.summary() )


def make_2Dfrom3D_quick(data, manyDz='all' ):
    print("Print using data with shape ",data.shape)
    #convert from 3D to 2D
    #stacking z cuts long time 
    assert  data.ndim == 5 #IS NOT A 3D dataset if assert fails
    ndZ = data.shape[3]
    ndT = data.shape[0]

    data = np.moveaxis(data, 3, 0)

    print("data.shape", data.shape)

    if manyDz=='all':
        lastDz = ndZ
        gapDz = 1
    else:
        lastDz = ndZ
        gapDz = lastDz//manyDz        
    print("lastDz, gapDz ",lastDz, gapDz)

    zIndxs = [0]
    for znum in range(1,lastDz,gapDz ):
        zIndxs.append(znum)

    print("Cutting Data 3d to 2d along z (may take a while . . .)")

    data = data[zIndxs]  

    print("zIndxs ", zIndxs)
    print("data.shape", data.shape)

    nz, nt, nx,ny, nch = data.shape

    data = data.reshape(nt* nz,  nx,ny, nch )
    print("Print returning data with shape ",data.shape)
    return data;
#dataM2 = make_2Dfrom3D_quick(data, manyDz=48 )
#dataM2 = make_2Dfrom3D_quick(data, manyDz=16 )

def loss_MSE_channelWise(y_pred, y_true):
    chnum = y_pred.shape[-1]
    loss =  loss_MSE(y_pred[...,0], y_true[...,0])
    print(loss)
    for ch in range(chnum-1):
        ll = loss_MSE(y_pred[...,ch], y_true[...,ch])
        loss = loss + ll
        print(ll)
    return loss;

def attention_block_MHSANew(x: tf.Tensor) -> tf.Tensor:
    print('x.shape at in ', x.shape)
    x = tf.keras.layers.LayerNormalization(axis=-1)(x)

    ##
    batch_size = x.shape[0]
    xorg_sh = x.shape
    #x = tf.reshape(x, (batch_size, xorg_sh[2],xorg_sh[3],xorg_sh[1]*xorg_sh[4]))
    x = tf.reshape(x, (batch_size, -1, xorg_sh[2],xorg_sh[3],xorg_sh[1]*xorg_sh[4]))
    print('x.shape at out s2d', x.shape)
    ##
    layerMHA = MultiHeadAttention(d_model=x.shape[-1], num_heads=x.shape[1]) #num_heads=8
    #output, attention_weights= temp_mha(x, k=x, q=x, mask=None) 

    q = AttentionVectorLayer()(x)
    v = AttentionVectorLayer()(x)
    k = AttentionVectorLayer()(x)

    output, attention_weights= layerMHA(v, k, q, mask=None) 

    print('output shape',output.shape )
    print('attention_weights shape',attention_weights.shape )
    return x + output

def attention_block_selfNew(x: tf.Tensor) -> tf.Tensor:
    print('x.shape at in ', x.shape)
    x = tf.keras.layers.LayerNormalization(axis=-1)(x)
    print('x.shape at out ', x.shape)
    
    q = AttentionVectorLayer()(x)
    v = AttentionVectorLayer()(x)
    k = AttentionVectorLayer()(x)

    print('q,v,k shape ', q.shape, v.shape, k.shape)
    output, attention_weights= scaled_dot_product_attention( q, k, v, None)
    print('output shape',output.shape )
    print('attention_weights shape',attention_weights.shape )
    return x + output


def attention_block(x: tf.Tensor) -> tf.Tensor:
    print('x.shape at in ', x.shape)
    x = tf.keras.layers.LayerNormalization(axis=-1)(x)
    print('x.shape at out ', x.shape)
    
    q = AttentionVectorLayer()(x)
    v = AttentionVectorLayer()(x)
    k = AttentionVectorLayer()(x)

    print('q,v,k shape ', q.shape, v.shape, k.shape)
    h = tf.keras.layers.Attention()([q, v, k])
    print('h shape',h.shape )
    return x + h

class AttentionVectorLayer(tf.keras.layers.Layer):
    """
    Building the query, key or value for self-attention
    from the feature map
    """
    def __init__(self, **kwargs):
        super(AttentionVectorLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        self.n_channels = input_shape[-1]
        self.w = self.add_weight(
            shape=(self.n_channels, self.n_channels),
            initializer=tf.keras.initializers.VarianceScaling(scale=1., mode='fan_avg', distribution='uniform'),
            trainable=True,
            name='attention_w'
        )
        self.b = self.add_weight(
            shape=(self.n_channels,),
            initializer='zero',
            trainable=True,
            name='attention_b'
        )

    def call(self, x: tf.Tensor) -> tf.Tensor:
        return tf.tensordot(x, self.w, 1) + self.b

    def get_config(self) -> dict:
        return {}

    @classmethod
    def from_config(cls, config):
        return cls(**config)


