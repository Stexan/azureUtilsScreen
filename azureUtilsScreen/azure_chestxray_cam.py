### Copyright (C) Microsoft Corporation.  

import keras.backend as K
import sys, os, io
import numpy as np
import cv2

import matplotlib
matplotlib.use('agg')

paths_to_append = [os.path.join(os.getcwd(), os.path.join(*(['Code',  'src'])))]
def add_path_to_sys_path(path_to_append):
    if not (any(path_to_append in paths for paths in sys.path)):
        sys.path.append(path_to_append)
[add_path_to_sys_path(crt_path) for crt_path in paths_to_append]

import azureUtilsScreen.azure_chestxray_utils as azure_chestxray_utils


def get_score_and_cam_picture(cv2_input_image, DenseNetImageNet121_model):
# based on https://github.com/jacobgil/keras-cam/blob/master/cam.py
    height, width , _ = cv2_input_image.shape

    class_weights = DenseNetImageNet121_model.layers[-1].get_weights()[0]
    #layer -3 without dropout
    final_conv_layer = DenseNetImageNet121_model.layers[-4]
    get_output = K.function([DenseNetImageNet121_model.layers[0].input],
                            [final_conv_layer.output, \
                             DenseNetImageNet121_model.layers[-1].output])
    [conv_outputs, prediction] = get_output([cv2_input_image[None,:,:,:]])
    conv_outputs = conv_outputs[0, :, :, :]
    prediction = prediction[0,:]

    #Create the class activation map.
    predicted_disease = np.argmax(prediction)
    cam = np.zeros(dtype = np.float32, shape = conv_outputs.shape[:2])
    #cam = np.zeros(dtype = np.float32, shape = conv_outputs.shape[1:3])
    print(conv_outputs.shape)
    print(cam.shape)
    print(class_weights.shape)
    print(class_weights[:,predicted_disease].shape)
    
    for i, w in enumerate(class_weights[:, predicted_disease]):
        cam += w * conv_outputs[:, :, i]

    return prediction, cam, predicted_disease

def process_cam_image(crt_cam_image, originalImage, crt_alpha = .6):
    xray_image = originalImage
    im_height, im_width, _ = xray_image.shape
    
    crt_cam_image /= np.max(crt_cam_image)
    crt_cam_image = cv2.resize(crt_cam_image, (im_width, im_height))

    heatmap = cv2.applyColorMap(np.uint8(255*crt_cam_image), cv2.COLORMAP_JET)

    heatmap[np.where(crt_cam_image < 0.2)] = 0

    #heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    
    print(heatmap.shape)
    print(xray_image.shape)
    print(crt_cam_image.shape)
    blendedImage = cv2.addWeighted(xray_image.astype('uint8'),0.8,\
                                   heatmap.astype('uint8'),(1-crt_alpha),0)
                                   
    return (blendedImage)

def plot_cam_results(crt_blended_image, crt_cam_image, crt_xray_image, map_caption):
    import matplotlib.pyplot as plt

    fig = plt.figure(figsize = (15,7))

    ax1 = fig.add_subplot(2, 3, 1)
    ax1.imshow(crt_xray_image, cmap = 'gray', interpolation = 'bicubic')
    ax1.set_title('Orig X Ray')
    plt.axis('off')

    ax2 = fig.add_subplot(2,3, 2)
    cam_plot = ax2.imshow(crt_cam_image, cmap=plt.get_cmap('OrRd'), interpolation = 'bicubic')
    plt.colorbar(cam_plot, ax=ax2)
    ax2.set_title('Activation Map')
    plt.axis('off')

    ax3 = fig.add_subplot(2,3, 3)
    blended_plot = ax3.imshow(crt_blended_image, interpolation = 'bicubic')
    plt.colorbar(cam_plot, ax=ax3)
    ax3.set_title(map_caption)
    plt.axis('off')
    
    # serialize blended image plot padded in the x/y-direction
    image_as_BytesIO = io.BytesIO()
    x_direction_pad = 1.05;y_direction_pad=1.2
    extent = ax3.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
    fig.savefig(image_as_BytesIO, 
                bbox_inches=extent.expanded(x_direction_pad, 
                                            y_direction_pad),
               format='png')
    image_as_BytesIO.seek(0)
    return(image_as_BytesIO)
    
def normalize(crt_array):
    crt_array /= 255.
    #mean and std are rgb
    #crt_array is bgr
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    
    crt_array[..., 0] /= np.std(crt_array[..., 0])
    crt_array[..., 0] *= std[2]
    
    crt_array[..., 0] -= np.mean(crt_array[..., 0])
    crt_array[..., 0] += mean[2]
    
    crt_array[..., 1] /= np.std(crt_array[..., 1])
    crt_array[..., 1] *= std[1]
    
    crt_array[..., 1] -= np.mean(crt_array[..., 1])
    crt_array[..., 1] += mean[1]
    
    crt_array[..., 2] /= np.std(crt_array[..., 2])
    crt_array[..., 2] *= std[0]
    
    crt_array[..., 2] -= np.mean(crt_array[..., 2])
    crt_array[..., 2] += mean[0]
    
    return crt_array
    
def process_xray_image(crt_xray_image, DenseNetImageNet121_model, originalImage):

    crt_xray_image=normalize(crt_xray_image)

    crt_predictions, crt_cam_image, predicted_disease_index = \
    get_score_and_cam_picture(crt_xray_image, 
                              DenseNetImageNet121_model)
    
    prj_consts = azure_chestxray_utils.chestxray_consts()
    likely_disease=prj_consts.DISEASE_list[predicted_disease_index]
    likely_disease_prob = 100*crt_predictions[predicted_disease_index]
    likely_disease_prob_ratio=100*crt_predictions[predicted_disease_index]/sum(crt_predictions)

    print('predictions: ', crt_predictions)
    print('likely disease: ', likely_disease)
    print('likely disease prob: ', likely_disease_prob)
    print('likely disease prob ratio: ', likely_disease_prob_ratio)
    
    probabilities = []
    probabilityRatios = []
    for value in crt_predictions:
        likely_disease_prob = 100*value
        likely_disease_prob_ratio=100*value/sum(crt_predictions)
        probabilities.append(likely_disease_prob)
        probabilityRatios.append(likely_disease_prob_ratio)
    
    crt_blended_image = process_cam_image(crt_cam_image, originalImage)
#    plot_cam_results(crt_blended_image, crt_cam_image, crt_xray_image,
#                    str(likely_disease)+ ' ' +
#                    "{0:.1f}".format(likely_disease_prob)+ '% (weight ' +
#                    "{0:.1f}".format(likely_disease_prob_ratio)+ '%)')

    dict = {'diseases': prj_consts.DISEASE_list, 'likelyIndex': predicted_disease_index, 'probabilities':probabilities, 'probabilityRatios': probabilityRatios}
    return dict

def process_nih_data(crt_image, DenseNetImageNet121_model):
    prj_consts = azure_chestxray_utils.chestxray_consts()
    
    crt_xray_image = cv2.resize(crt_image,
                                (prj_consts.CHESTXRAY_MODEL_EXPECTED_IMAGE_HEIGHT,
                                 prj_consts.CHESTXRAY_MODEL_EXPECTED_IMAGE_WIDTH)) \
                    .astype(np.float32)

    resultDict = process_xray_image(crt_xray_image, DenseNetImageNet121_model, crt_image)

    return resultDict
        
if __name__=="__main__":
    #FIXME
    # add example/test code here
    
    NIH_annotated_Cardiomegaly = ['00005066_030.png']
    data_dir = ''
    cv2_image = cv2.imread(os.path.join(data_dir,NIH_annotated_Cardiomegaly[0]))

    print_image_stats_by_channel(cv2_image)
    cv2_image = normalize_nd_array(cv2_image)
    cv2_image = 255*cv2_image
    cv2_image=cv2_image.astype('uint8')
    print_image_stats_by_channel(cv2_image)

    predictions, cam_image, predicted_disease_index = get_score_and_cam_picture(cv2_image, model)
    print(predictions)
    prj_consts = azure_chestxray_utils.chestxray_consts()
    print(prj_consts.DISEASE_list[predicted_disease_index])
    print('likely disease: ', prj_consts.DISEASE_list[predicted_disease_index])
    print('likely disease prob ratio: ', \
          predictions[predicted_disease_index]/sum(predictions))
    blended_image = process_cam_image(cam_image, cv2_image)
    plot_cam_results(blended_image, cam_image, cv2_image, \
                 prj_consts.DISEASE_list[predicted_disease_index])
