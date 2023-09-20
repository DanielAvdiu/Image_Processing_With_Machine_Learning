# Image_Processing_With_Machine_Learning
Image processing using machine learning. The machine learning models are used for cell image segmentation. The main structure of the model is U-Net, accompanied with some interesting variations.


The initial ideas was to use just the U-Net architecture to build the machine learning model, however this method would have not provided us with an innovative approach. While using the U-Net architecture it was seen that the segmentation was performed good enough, with the exception that more improvement was needed. Then I used some modifications to the original U-Net architecture, some of them being optimized for performance and the others being optimized for memory. On their own, all the alterations lacked a complementary part. 

<b> Pruning U-Net </b> (eliminating unnecessary weights on the filters) turned out to result in non-satisfactory results, with the exception that in terms of memory it was one of the best. Attention U-Net provided better results with the drawback of producing models that were bigger in size. The implementation of Attention U-Net is quite interesting, we just add attention gates in the skip connections between the encoding and decoding paths. It basically puts more emphasis in some of the weights that seem to be more important, and it decides this importance based on <b> Activation Functions </b>.
