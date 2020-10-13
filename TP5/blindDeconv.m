%blind deconvolution implementation
%Authors: Matías Larroque, Manuel Mollón, Tomás Gonzalez - ITBA 2020
%First we import the image to recover.
file_name_in = 'image_in_deconvblind.bmp';
blurred_image = imread(file_name_in);

%We have to estimate an initial PSF
initial_PSF = ones(12, 12)
%Call deconv function, which will estimate the best PSF possible and do de
%recovery.
image_recovered = deconvblind(blurred_image, initial_PSF);
%Save result
file_name_out = 'image_out_deconvblind.bmp';
imwrite(image_recovered, file_name_out);

