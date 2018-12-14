# Object_Detection_Enhancement_with_RBG-D_Data
EK500_final  project
<br>
Many image classification algorithms are based entirely on red-blue-green (RGB) images. These RGB images provide the appearance and texture of the object(s) to be classified. While RGB image classifiers present impressive results, adding depth information, in the form of RGB-D images, to the classifier yields even better results because it gives robust information about the shape, size and boundaries of the object(s). Our networks are based on  two separate networks, one for RGB image and one for depth image, both of which are joined together in the final layers.
To implement lighter networks, two models are built. For both of them only one channel for depth image is used as input.   The second network is not in parallel.  Instead, to include the depth information a fourth channel is combined with the three channels of RGB image.

Idea Based on the paper : https://arxiv.org/pdf/1507.06821.pdf
Multimodal Deep Learning for Robust RGB-D Object Recognition

Eitel, A., Springenberg, J. T., Spinello, L., Riedmiller, M., & Burgard, W. (2015, September). Multimodal deep learning for robust rgb-d object recognition. In Intelligent Robots and Systems (IROS), 2015 IEEE/RSJ International Conference on(pp. 681-687). IEEE.
