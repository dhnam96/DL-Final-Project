# DL-Final-Project
DL Final Project

http://bamos.github.io/2016/08/09/deep-completion/

Paper
https://arxiv.org/pdf/1904.07475.pdf

Transformer
https://arxiv.org/pdf/1802.05751.pdf

Database
https://www.kaggle.com/jessicali9530/stanford-cars-dataset


Pyramid Context Encoder Network:
-	Main point: pyramid context encoder that learns progressively the region affinity by attention from high-level semantic feature map. The encoder also transfers the learned attention to the previous low-level feature map.
-	Attention transfer from deep to shallow in pyramid fashion. Also incorporates multi-scale decoder with deeply-supervised pyramid loss and adversarial loss. 
-	Tl;dr : faster network training and more realistic results. Superior performance and faster computation.

We need to generate content that is both visually-realistic and semantically reasonable. 

Approach 1 – texture synthesis techniques to fill regions at image level
	Drawback: lack of high-level understanding of an image often results in failures in generating semantically-reasonable results.
	
Approach 2 – encode semantic context of an image into a latent feature space by deep neural networks and then generate a semantic-coherent patches by generative models.

Paper’s approach – generate visual and semantic coherence by filling regions at both image and feature levels. U-Net structure (encode context from low-level pixels to high-level semantic features and decode the features back into image)

Pyramid context: pyramid-context encoder, multiscale decoder, adversarial training loss (boost image inpainting)
Once encoded from image, the pyramid-context encoder fills regions from high-level semantic features to low-level features. Attention transfer network to learn region affinity between patches inside/outside missing regions in a high-level feature map, then transfer relevant features from outside to inside regions of previous feature map.
Takes input the reconstructed features from ATNs through skip connections and the latent features for final decoding. 

Note:
Cross-layer attention transfer

https://github.com/affinelayer/pix2pix-tensorflow
