# ml-linear-mapping
Assignment for CoE197M class (Machine Learning) last Nov 2022 wherein projective distortion was removed in images

Learning Objective:
1) Learn how to apply linear mapping concepts on removing projective distortion

Problem:
When an image is taken by the camera, there is an inherent projective distortion. This makes rectangular objects appear distorted. However, this projective distortion can be reversed if we know the 3x3 matrix that maps from projective to affine (frontal no distortion). 

Using our knowledge on linear mapping and least squares estimation, develop a program that will remove the projective distortion on a given image. Note that at least 4 pts on the target undistorted image must be known.
