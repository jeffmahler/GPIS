function H = high_res_surface(I, scale, win, sig)

if nargin < 3
    win = 10;
end
if nargin < 4
    sig = 3;
end

J = imresize(I, scale);
G = fspecial('gaussian', [win, win], sig);
H = imfilter(J, G, 'replicate');
H = imsharpen(H, 'Amount', 10);

H = 1 - (H < 0.8);

end

