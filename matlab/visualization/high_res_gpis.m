function H = high_res_gpis(I, scale)

J = imresize(I, scale);
G = fspecial('gaussian', [5, 5], 1.0); % note, this won't work well if scale much larger than 4
K = imfilter(J, G, 'replicate');
H = imsharpen(K, 'Amount', 10);

end

