from subprocess import call
import numpy as np
import os
import caffe
import shutil
#from keras.models import load_model
#from PIL import  Image

curr_dir = os.path.dirname(os.path.realpath(__file__))
model_dir = curr_dir + '/../../models/caffe/'
result_dir = curr_dir + '/../../result/png/'
data_dir = curr_dir + '/../../data/png/'

def violence_adult_scores(vid_id_list):
    if len(vid_id_list) == 0:
        print("video id's list is empty")
        return
    nsfw_net_yahoo = caffe.Net(model_dir + 'deploy_yahoo.prototxt',
                               model_dir + 'resnet_50_1by2_nsfw.caffemodel', caffe.TEST)
    nsfw_net_deep_miles = caffe.Net(model_dir + 'deploy_dm.prototxt',model_dir +'_iter_2000_dm.caffemodel', caffe.TEST)
    nsfw_net_2k = caffe.Net(model_dir + 'deploy_dm_2k.prototxt', model_dir + '_iter_1000_dm_2k.caffemodel' , caffe.TEST)
    adult_vid_scores = open(result_dir + "anuj_b2/adult_scores_p1_1128.txt", "a")
    ft_error = open(result_dir + "anuj_b2/error_p1_1128.txt", "a")
    for vid_id in vid_id_list:
        vid_id = vid_id.strip()
        #'''
        try:
            vid_url = 'www.youtube.com/watch?v=' + vid_id
            out_dir_videos = data_dir + 'anuj_b1/downloaded_videos/'
            out_dir_frames = data_dir + 'anuj_b1/video_frames/' + vid_id
            command = "youtube-dl -f 'bestvideo[height<=480]' -o " + out_dir_videos + "%s.mp4 %s" % (vid_id, vid_url)
            #print(command)
            call(command, shell=True)
            
            if not os.path.exists(out_dir_frames):
                os.makedirs(out_dir_frames)
            command_2 = "ffmpeg -i {0} -vf fps=1 {1}img%03d.jpg".format(out_dir_videos + vid_id + '.mp4',
                                                                    out_dir_frames + '/')
            call(command_2, shell=True)
            call("rm  {0}".format(out_dir_videos+vid_id + '.mp4'),shell=True)
            #'''
            #out_dir_frames = data_dir + '/test/' + vid_id
            get_yahoo_score(out_dir_frames + '/', nsfw_net_yahoo, nsfw_net_deep_miles, nsfw_net_2k, adult_vid_scores)
            #adult_vid_score.write(vid_id+','+res+'\n')
        except Exception as e:
            print(e)
            ft_error.write(vid_id+'\n')    
    adult_vid_scores.close()
    ft_error.close()    


def caffe_preprocess_and_compute(pimg, caffe_transformer=None, caffe_net=None,
                                 output_layers=None):
    """
    Run a Caffe network on an input image after preprocessing it to prepare
    it for Caffe.
    :param PIL.Image pimg:
        PIL image to be input into Caffe.
    :param caffe.Net caffe_net:
        A Caffe network with which to process pimg afrer preprocessing.
    :param list output_layers:
        A list of the names of the layers from caffe_net whose outputs are to
        to be returned.  If this is None, the default outputs for the network
        are returned.
    :return:
        Returns the requested outputs from the Caffe net.
    """
    if caffe_net is not None:

        # Grab the default output names if none were requested specifically.
        if output_layers is None:
            output_layers = caffe_net.outputs

        image = caffe.io.load_image(pimg)

        H, W, _ = image.shape
        _, _, h, w = caffe_net.blobs['data'].data.shape
        h_off = int(max((H - h) / 2, 0))
        w_off = int(max((W - w) / 2, 0))
        crop = image[h_off:h_off + h, w_off:w_off + w, :]
        transformed_image = caffe_transformer.preprocess('data', crop)
        transformed_image.shape = (1,) + transformed_image.shape

        input_name = caffe_net.inputs[0]
        all_outputs = caffe_net.forward_all(blobs=output_layers,
                                            **{input_name: transformed_image})

        outputs = all_outputs[output_layers[0]][0].astype(float)
        return outputs
    else:
        return []


def get_yahoo_score(dir_name, nsfw_net_yahoo, nsfw_net_deep_miles, nsfw_net_2k, ft_main=None):
    caffe_transformer = caffe.io.Transformer({'data': nsfw_net_yahoo.blobs['data'].data.shape})
    caffe_transformer.set_transpose('data', (2, 0, 1))  # move image channels to outermost
    caffe_transformer.set_mean('data', np.array([104, 117, 123]))  # subtract the dataset-mean value in each channel
    caffe_transformer.set_raw_scale('data', 255)  # rescale from [0, 1] to [0, 255]
    caffe_transformer.set_channel_swap('data', (2, 1, 0))  # swap channels from RGB to BGR

    _total = 0
    _dir = dir_name.strip().split('/')[-2].strip()
    print("video_directory",_dir)
    res_dir = result_dir + 'anuj_b2/p1_1128/'
    if not os.path.exists(res_dir + str(_dir)):
        os.makedirs(res_dir + _dir)
        os.makedirs(res_dir + _dir + "/normal")
        os.makedirs(res_dir + _dir + '/adult_high')
        os.makedirs(res_dir + _dir + '/adult_low')
    nm_file = open(res_dir + str(_dir)+'/normal_result.txt', 'a')

    for _file in os.listdir(dir_name):
        image_data = dir_name + '/' + _file
        scores_yahoo = caffe_preprocess_and_compute(image_data, caffe_transformer=caffe_transformer,
                                                    caffe_net=nsfw_net_yahoo,
                                                    output_layers=['prob'])
        scores_p2k = caffe_preprocess_and_compute(image_data, caffe_transformer=caffe_transformer,
                                                    caffe_net=nsfw_net_2k,
                                                    output_layers=['prob'])
        scores_deep_miles = caffe_preprocess_and_compute(image_data, caffe_transformer=caffe_transformer,
                                                    caffe_net=nsfw_net_deep_miles,
                                                    output_layers=['prob'])
        
        value_yahoo = float(scores_yahoo[1])
        value_p2k = float(scores_p2k[1])
        value_deep_miles = float(scores_deep_miles[1])
        score_total = str(_dir)+",yh_"+str("{:.2f}".format(value_yahoo)) + ",p2k_"+str("{:.2f}".format(value_p2k)) + ",d_m_"+str("{:.2f}".format(value_deep_miles))+","+str(_file)
        print (score_total)
        ft_main.write(score_total+'\n')
        # max_age = scores.argmax(axis=0)
        print(value_yahoo, _dir)

        _count_ensemble = 0
        if value_yahoo > 0.40:
            _count_ensemble += 1
        _total += 1
        if value_p2k > 0.60:
            _count_ensemble += 1
        _total += 1
        if value_deep_miles > 0.50:
            _count_ensemble += 1
        _total += 1

        if _count_ensemble >=3 :
            shutil.copy(image_data,
                        res_dir + _dir +  "/adult_high/" + score_total)
        elif  _count_ensemble >= 1:
            shutil.copy(image_data,
                        res_dir + _dir +  "/adult_low/" + score_total)
        else :
            nm_file.write(dir_name + '/' + _file+','+score_total+'\n')
            #shutil.copy(image_data,
            #            res_dir + _dir +  "/normal/" + score_total)
    nm_file.write.close()

png_batch_new_adult_1 = [ 'ztZxpHlZJmU','hs26KtgKSFw','hw0NdZPOQSM','9FsofD_Y0H0','InTXT-t8gmE','bOw8E9GwNq8','XYF74Y4ok6U','-GrOEpFi4jc','csqE3An9CZk','LvLjb_25CSU','6cAsGh6Hhio','eWbxrPMW-Yc','nVFUiFVB_GU','pb0ApjoJKHo','5nd_dOG8WJM','0UgkwVdZEns','W7n1nCxd-i0','g6bysZiH0xI','VFxchHwl5p4','Lr-3XK5Y4Hc','hSTEDgxyaHU','Vl9O4aeaLoQ','i4-5oADYYwc','WGh7sQwvqMY','EtMpsB9t9Zo','7vAj9nBiZqg','L-Ib2g3-hr0','90GYJG-qk3g','KN8R4rei1rI','eELK4KNkh0E','Lh7TjRXXYsg']
png_list_1 = open('anuj_b1_p1_1128_vid.txt', 'r').readlines()
png_list_2 = open('anuj_b1_p2_2024_vid.txt', 'r').readlines()
violence_adult_scores(png_list_1)
