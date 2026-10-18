# CONCEPTS: EST Project (Steel Surface Defect CNN)

Every concept we use, explained simply. The format is in `CONTRIBUTING.md` section 4.7.
Add a concept the first time it is used. Improve an entry when our understanding improves.

## Domain (steel & inspection)

### Hot rolling
- **In one line:** Squeezing a heated steel slab between rolls to make thin strip.
- **Explained simply:** A slab is heated to roughly 1100 to 1250 degrees C, then passed through rolling mills that make it thinner in steps. "Hot" means the steel is rolled above the temperature at which its grains re-form, so it stays soft enough to shape. The strip is then cooled with water and wound into a coil. Think of rolling out dough, but with steel and much more force.
- **Formula (if any):** None.
- **Why we used it here:** All NEU images are surfaces of hot-rolled strip, and the defects are tied to steps in this process (descaling, rolls).
- **Where in our code:** Not in code. Background in `docs/problem.md`.
- **Common confusion / trap:** Hot rolling is not cold rolling. Cold rolling is done at room temperature afterwards and gives a smoother finish. NEU is hot-rolled strip only.

### Crazing
- **In one line:** The NEU class that shows a network of fine surface cracks.
- **Explained simply:** Crazing images show many thin cracks spread over the surface, like the cracks in dry mud or old paint. In our project it is the "crack" class, because the problem statement asks about cracks.
- **Formula (if any):** None.
- **Why we used it here:** It lets us answer "identify cracks" directly: besides the 6-class results, we report precision, recall and F1 for crazing against the rest.
- **Where in our code:** Class label `crazing` (code `Cr`). Crack-vs-rest metrics are computed in the evaluation phase.
- **Common confusion / trap:** Some dataset mirrors call this class "cracks". We did not find a source for what causes crazing, so we do not claim one. It looks similar to rolled-in scale, so the two may be confused.

### Rolled-in scale
- **In one line:** Oxide scale pressed into the strip surface during rolling.
- **Explained simply:** Hot steel reacts with air and grows a crusty oxide layer called scale. Water sprays (descaling) wash it off before rolling. Scale that survives gets squashed into the strip by the rolls and leaves rough marks. Our sources link it to low spray pressure, blocked nozzles and uneven furnace temperature.
- **Formula (if any):** None.
- **Why we used it here:** It is our main example of predictive maintenance: more rolled-in scale suggests checking the descaling system.
- **Where in our code:** Class label `rolled-in_scale` (code `RS`).
- **Common confusion / trap:** A rising count is a hint to inspect, not proof of a faulty descaler, since one defect can have several causes. It can also be mistaken for hot-strip scratches.

### Line-scan camera
- **In one line:** A camera with a single row of pixels that builds an image as the object moves past.
- **Explained simply:** A normal camera takes a whole picture at once. A line-scan camera reads one thin line, again and again, while the strip moves under it, and stacks the lines into a long image. So the image can be as long as the strip needs, and as wide as the sensor.
- **Formula (if any):** None.
- **Why we used it here:** It explains why a real line gives long, high-resolution images, while NEU gives 200 x 200 crops. A real system would cut the long image into tiles and classify each one.
- **Where in our code:** Not in code. Possible tiling demo in the extras phase.
- **Common confusion / trap:** NEU images are not raw line-scan output. How they were cropped is our assumption, not something we verified.

### Automated visual inspection
- **In one line:** Cameras plus software that find and classify defects without a human looking.
- **Explained simply:** Cameras photograph the strip as it moves, and a program decides whether each image shows a defect and which type. Unlike a tired inspector, it applies the same rule to every image. Our CNN is the "decision" part of such a system.
- **Formula (if any):** None.
- **Why we used it here:** It is the setting of the problem statement: Tata Steel wants to automate quality inspection.
- **Where in our code:** The whole project is a prototype of the classifier step.
- **Common confusion / trap:** Classification (what defect is in this image) is not detection (where in the image). We do classification, and use Grad-CAM later to see where the model looked.

## Images & pre-processing

### Digital image as a matrix of pixels
- **In one line:** A grayscale image is a grid of numbers, one number per pixel.
- **Explained simply:** A pixel is one tiny square of the picture. Its number says how bright it is: 0 is black and 255 is white. Our 200 x 200 image is a table with 200 rows and 200 columns, so 40,000 numbers. A computer never "sees" a defect. It only sees this table, and the CNN learns patterns in it.
- **Formula (if any):** Grayscale image shape = height x width. Colour image shape = height x width x 3.
- **Why we used it here:** Every step in the project (hashing, averaging, PCA, convolution) is arithmetic on these tables.
- **Where in our code:** Notebook 02, section 'The split' onwards: `images` is a NumPy array of shape (number of images, 200, 200).
- **Common confusion / trap:** Row comes before column, so shape (200, 200) is (height, width). A pixel value of 255 is not "high quality". It just means white.

### Grayscale vs RGB
- **In one line:** Grayscale has one number per pixel (brightness). RGB has three (red, green, blue).
- **Explained simply:** A colour pixel stores how much red, green and blue it has, so a colour image has 3 channels. A grayscale image has 1 channel. If all three channels of a colour file are equal, the picture has no colour information at all. It is a grayscale picture saved in a colour format.
- **Formula (if any):** None.
- **Why we used it here:** All 1,800 NEU files are stored as RGB, but in every one the three channels are identical. So we load them as 1 channel for our own CNN (200 x 200 x 1). MobileNetV2 was trained on colour photos and expects 3 channels, so for it we repeat the one channel three times (224 x 224 x 3).
- **Where in our code:** Notebook 01, sections 'Size, colour mode and file format' and 'What we found'. Notebook 03, section 'The pipeline' (`channels=1`, then `grayscale_to_rgb`).
- **Common confusion / trap:** Repeating a channel adds no new information. It only makes the shape fit the pretrained model.

### Image resolution
- **In one line:** The number of pixels in an image, written width x height.
- **Explained simply:** More pixels means more detail but also more memory and slower training. All our images are 200 x 200 px. A real line-scan camera gives much larger images, so a real system would cut them into tiles of this size.
- **Formula (if any):** Number of pixels = width x height. Here 200 x 200 = 40,000.
- **Why we used it here:** The scratch CNN takes 200 x 200 directly. MobileNetV2 is fed 224 x 224, so we resize up by a small amount.
- **Where in our code:** Notebook 01, section 'Size, colour mode and file format'. Notebook 03, `tf.image.resize` in `make_dataset`.
- **Common confusion / trap:** Resolution is not the same as image quality. Resizing a small image up does not add real detail.

### File hash (MD5) and duplicate detection
- **In one line:** A hash is a short fingerprint of data, and two identical inputs always give the same fingerprint.
- **Explained simply:** MD5 reads all the bytes of a file and produces a 32-character code. Change one byte and the code changes completely. We compute two fingerprints per image: `file_md5` of the file bytes, and `pixel_md5` of the decoded pixels. The second one also catches the same picture saved as two different files. Two images with the same fingerprint are duplicates.
- **Formula (if any):** None.
- **Why we used it here:** We found one duplicate pair: `patches_101.jpg` and `patches_105.jpg` are byte-identical, and both are labelled patches. Both hashes caught it.
- **Where in our code:** Notebook 01, section 'Find every image' (computing hashes) and section 'Exact duplicates' (grouping).
- **Common confusion / trap:** MD5 finds exact copies only. A slightly cropped or re-saved copy gets a different fingerprint. MD5 is fine for duplicate checks, but it is not safe for security.

### Stratified split
- **In one line:** A split that keeps each class's share the same in train, validation and test.
- **Explained simply:** If we split 1,800 images at random, a class could end up with few images in the test set by bad luck. Stratified means we split inside each class, so every part has the same mix. We use 70 % train, 15 % validation, 15 % test with seed 42. That gives 210 / 45 / 45 images per class and 1,260 / 270 / 270 in total.
- **Formula (if any):** Per class: 300 x 0.70 = 210, 300 x 0.15 = 45.
- **Why we used it here:** It makes every part a fair mini copy of the whole dataset, so test accuracy is not skewed by one class being rare.
- **Where in our code:** Notebook 02, section 'The split'.
- **Common confusion / trap:** Stratify on the class label, not on the file name. The seed only makes the split repeatable. It does not make it better.

### Group-aware split
- **In one line:** A split that moves a whole group of linked samples into the same part.
- **Explained simply:** Some samples are not independent. Identical images are the clearest case. If one copy lands in train and the other in test, the model has already seen the test image. So we first group images that have identical pixels, then split the groups, not the single images. We stratify these groups by class.
- **Formula (if any):** None.
- **Why we used it here:** `patches_101.jpg` and `patches_105.jpg` form one group of two, so they always stay together. After the split, 0 pictures appear in two parts, and an `assert` checks this.
- **Where in our code:** Notebook 02, section 'The split' (grouping by `pixel_md5`) and the safety check below it.
- **Common confusion / trap:** A group-aware split can make the part sizes differ by a few images from the exact percentages. Group ids are not the same as class labels.

### Data leakage (for images)
- **In one line:** Information from the test set sneaks into training, so the test score looks better than it is.
- **Explained simply:** The test set should stand in for images the model has never seen. It stops doing that if a test image, or a near-copy of one, was used in training. For images, the usual causes are duplicates across parts, augmented copies of one image in both train and test, and looking at test images while designing the model.
- **Formula (if any):** None.
- **Why we used it here:** We split before any exploration or augmentation, split by groups of identical pixels, and never augment validation or test images. Notebook 03 ends with a leakage checklist.
- **Where in our code:** Notebook 02, intro ('Why split first') and 'The split'. Notebook 03, section 'Leakage checklist'.
- **Common confusion / trap:** Leakage gives no error message. The only sign is a score that looks too good.

### Mean image
- **In one line:** The average of many images, pixel by pixel.
- **Explained simply:** Add up all training images of one class and divide by their number. Random details cancel out. What stays is what the class has in common, such as a direction or uneven lighting. A flat grey mean image means the class has no fixed position or orientation.
- **Formula (if any):** mean image at pixel (i, j) = (1/N) x sum over all images of the pixel value at (i, j).
- **Why we used it here:** In the means of crazing, inclusion, pitted surface and scratches, the image darkens towards the right edge. This points to uneven lighting from the camera setup. The scratches mean shows both vertical and horizontal lines.
- **Where in our code:** Notebook 02, section 'EDA 3'.
- **Common confusion / trap:** The mean image is a summary, not a real sample. A clean pattern in it can come from the lighting rig, not from the defect.

### Pixel intensity histogram
- **In one line:** A bar chart of how many pixels have each grey level from 0 to 255.
- **Explained simply:** For one class we count all pixels at value 0, at value 1, and so on up to 255. A peak on the left means many dark pixels. A peak on the right means many bright ones. If the curves of two classes lie on top of each other, brightness alone cannot tell them apart.
- **Formula (if any):** None.
- **Why we used it here:** Our class histograms overlap heavily, and the brightness ranges overlap across classes. So a simple brightness rule is not enough. Patches and pitted surface have a spike at 255, which means over-exposed (washed-out white) parts.
- **Where in our code:** Notebook 02, section 'EDA 4'. Per-image brightness (mean pixel value) and contrast (pixel standard deviation) are in section 'EDA 5'.
- **Common confusion / trap:** A histogram ignores where the pixels are, so two very different images can have the same histogram. Median brightness by class: crazing 148.67, inclusion 99.74, patches 139.12, pitted surface 186.05, rolled-in scale 126.53, scratches 90.36. Patches have the highest median contrast (57.75).

### PCA (principal component analysis)
- **In one line:** A method that squeezes many numbers per sample into a few, keeping as much variation as possible.
- **Explained simply:** Each 50 x 50 image is 2,500 numbers, one per pixel. PCA finds the direction in which the images differ most (PC1), then the next most (PC2), and so on. Plotting every image by its PC1 and PC2 gives a 2-D map. Images that look alike land near each other. The percentage tells how much of the total variation each direction holds.
- **Formula (if any):** None needed here. PC1 explained variance ratio = variance along PC1 / total variance.
- **Why we used it here:** PC1 holds 70.63 % of the variance and PC2 only 8.28 %. In the plot, pitted surface sits to the right, scratches to the left, and crazing, inclusion and rolled-in scale overlap in the middle. So raw pixels do not separate all six classes, and we need a model that learns its own features.
- **Where in our code:** Notebook 02, section 'EDA 6' (`PCA(n_components=2)` on training images shrunk to 50 x 50).
- **Common confusion / trap:** PCA does not use the labels. Colours in the plot are added afterwards. Overlap in two components does not prove a classifier cannot separate the classes. It only shows raw pixels are not enough for a simple view. In our data PC1 is simply overall brightness: its correlation with each image's mean pixel value is 1.0 (notebook 02, `eda_summary.json`).

### Normalisation (0 to 1 vs -1 to 1, `preprocess_input`)
- **In one line:** Rescaling pixel values to a small, fixed range before they enter the network.
- **Explained simply:** Raw pixels run from 0 to 255. Networks train better on small numbers. Our scratch CNN divides by 255, giving 0 to 1. MobileNetV2 was pretrained on images scaled to -1 to 1, and `mobilenet_v2.preprocess_input` does exactly that. If we fed it 0 to 1 values instead, nothing would crash, but it would train badly.
- **Formula (if any):** 0 to 1: x / 255. -1 to 1: x / 127.5 - 1.
- **Why we used it here:** Each model must get the scale it expects.
- **Where in our code:** Notebook 03, section 'The pipeline' (`img / 255.0` and `preprocess_input` inside `load`).
- **Common confusion / trap:** The scale must be the same at training and test time. Normalising is not the same as augmenting.

### Data augmentation
- **In one line:** Showing the model randomly changed copies of the training images.
- **Explained simply:** We only have 210 training images per class, so a CNN could memorise them. Each epoch, we randomly flip, tilt, zoom and change the contrast and brightness of every image a little. The model sees a new version each time, so it has to learn what the defect looks like. Our settings: flip horizontal and vertical, rotation 0.05 (about plus or minus 18 degrees) with reflect fill, zoom 0.1, contrast 0.2, brightness 0.2. We avoid 90 degree turns, because scratches and rolled-in scale follow the rolling direction.
- **Formula (if any):** None. RandomRotation(0.05) means 0.05 of a full turn, so 0.05 x 360 = 18 degrees.
- **Why we used it here:** It fights overfitting on a small dataset. Only training images are augmented. Validation and test images must stay as they are, because they stand in for real, unseen images, and we want an honest score. The layers are Keras layers placed inside the model, and Keras switches them on only in training mode. So the same model automatically skips augmentation when predicting, and nobody can forget to turn it off.
- **Where in our code:** Notebook 03, sections 'Augmentation' and 'Leakage checklist'.
- **Common confusion / trap:** Augmentation does not add new real images. It also needs a physical reason. A change that never happens on the line only teaches the model something false.

### Batch
- **In one line:** A small group of images processed together in one training step.
- **Explained simply:** We do not feed all 1,260 images at once, and we do not feed one at a time. We feed 32 at a time. The model computes the loss on those 32, updates its weights once, then takes the next 32. Batches fit in GPU memory and give a steadier update than one image.
- **Formula (if any):** Steps per epoch = ceil(images / batch size). Here ceil(1260 / 32) = 40.
- **Why we used it here:** Batch size is 32 for all models.
- **Where in our code:** Notebook 03, section 'The pipeline' (`BATCH = 32`).
- **Common confusion / trap:** A batch is not an epoch. An epoch is one pass over all training images, so it is many batches. The last batch can be smaller than 32.

### `tf.data` pipeline, `cache` and `prefetch`
- **In one line:** TensorFlow's way to load, transform and batch data efficiently while the model trains.
- **Explained simply:** The pipeline starts from file paths and labels, shuffles, reads and decodes each JPEG, resizes it, scales the pixels, groups images into batches of 32, and hands them to the model. `map` with `num_parallel_calls` runs the loading on several CPU threads. `prefetch` prepares the next batch while the GPU is busy with the current one, so the GPU never waits. `cache` keeps decoded images in memory after the first epoch so they are not decoded again.
- **Formula (if any):** None.
- **Why we used it here:** It keeps the GPU on Kaggle busy. Notebook 03 builds the pipeline with `prefetch` only (it reads just a few batches). Notebook 04, which trains for many epochs, adds `cache` after decoding, so each image is decoded once; the training set is shuffled after the cache, so the order still changes every epoch.
- **Where in our code:** Notebook 03, section 'The pipeline' (`make_dataset`).
- **Common confusion / trap:** Never put random augmentation before `cache`, or the random changes are frozen after the first epoch. Shuffle only the training data.

## CNN building blocks

### Convolution (kernel, stride, padding, feature map)
- **In one line:** A small filter that slides over the image and marks where its pattern appears.
- **Explained simply:** The filter (also called a kernel) is a tiny grid of numbers, 3 × 3 in our models. We place it on one patch of the image, multiply each pixel by the matching filter number and add everything up. That gives one number. Then we slide the filter along and repeat. All the numbers together form a feature map: a new image that is bright where the filter's pattern (an edge, a texture) was found. Stride is how many pixels the filter jumps each time. Padding adds a border of zeros so the filter can also sit on the edge pixels. The filter numbers are the weights the network learns.
- **Formula (if any):** Output value = sum of (pixel × filter weight) over the patch, plus a bias. Parameters of one layer = (3 × 3 × input channels + 1) × number of filters. Layer 1 of M1: (3 × 3 × 1 + 1) × 32 = 320.
- **Why we used it here:** It finds local patterns such as cracks, pits and scratches wherever they sit in the image, with far fewer weights than a layer connected to every pixel.
- **Where in our code:** Notebook 04, section 'M1' (`Conv2D(filters, 3, padding="same")`, filters 32, 64, 128). Stride is the Keras default of 1.
- **Common confusion / trap:** `padding="same"` keeps the image size the same (200 × 200 stays 200 × 200). It is the pooling layer that shrinks it. One filter gives one feature map, so 32 filters give 32 maps.

### ReLU
- **In one line:** An activation that keeps positive numbers and turns negative numbers into 0.
- **Explained simply:** After each convolution we pass every number through ReLU. If the number is below zero it becomes zero, otherwise it stays. Without this step, stacking layers would still behave like one straight-line calculation. The bend at zero lets the network learn curved, complicated rules. It is also cheap to compute.
- **Formula (if any):** ReLU(x) = max(0, x).
- **Why we used it here:** It is the standard choice for CNNs and trains fast. Every convolution and the first dense layer in our models use it.
- **Where in our code:** Notebook 04, section 'M1' (`Activation("relu")`, `Dense(128, activation="relu")`).
- **Common confusion / trap:** ReLU has no weights to learn. In M2 it comes after batch normalisation, so the order is convolution, batch norm, ReLU.

### Max pooling
- **In one line:** Shrinks a feature map by keeping only the strongest value in each small block.
- **Explained simply:** We cut the feature map into 2 × 2 blocks and keep the largest number from each block. The map becomes half as wide and half as tall. The strongest response survives, so the network still knows the pattern was there, but it cares less about exactly where. The next layer also has less work to do.
- **Formula (if any):** Output size = input size / 2 for a 2 × 2 pool. In M1: 200 to 100 to 50 to 25 pixels.
- **Why we used it here:** After each of the three convolution blocks it halves the image while the number of filters doubles (32, 64, 128), so later layers see larger patterns.
- **Where in our code:** Notebook 04, section 'M1' (`MaxPooling2D(2)`).
- **Common confusion / trap:** Pooling has no weights. It also does not change the number of feature maps, only their size.

### Global average pooling (GAP)
- **In one line:** Replaces each whole feature map by its average, one number per map.
- **Explained simply:** After the last pooling layer M1 has 128 feature maps of 25 × 25. GAP averages each map into a single number, so we get a list of 128 numbers. Each number says how strongly one learned pattern appeared anywhere in the image. The classifier then works on this short list.
- **Formula (if any):** Output of map k = average of all 25 × 25 values in map k.
- **Why we used it here:** Flattening the maps would give 25 × 25 × 128 = 80,000 numbers and a huge dense layer that can memorise 1,260 images. GAP gives 128 numbers and adds no weights. It is a main reason M1 has only 109,958 parameters. MobileNetV2's head uses it too.
- **Where in our code:** Notebook 04, section 'M1'; notebook 05, section 'M3' (`GlobalAveragePooling2D()`).
- **Common confusion / trap:** GAP throws away where in the image a pattern was found. For texture defects that is acceptable.

### Dense layer
- **In one line:** A layer where every input number is connected to every output number.
- **Explained simply:** Each output is a weighted sum of all inputs plus a bias, followed by an activation. It combines the patterns found by the convolution layers into a decision. M1 has two: a hidden one with 128 units, then an output one with 6 units, one per defect class.
- **Formula (if any):** Parameters = (inputs + 1) × units. Hidden: (128 + 1) × 128 = 16,512. Output: (128 + 1) × 6 = 774.
- **Why we used it here:** The last layer must produce 6 class scores. The hidden layer lets the network mix the 128 pattern strengths before deciding.
- **Where in our code:** Notebook 04, section 'M1' (`Dense(128)` and `Dense(len(CLASSES))`); notebook 05, section 'M3' (one `Dense` for 6 classes).
- **Common confusion / trap:** In many CNNs the dense layers hold most of the weights. That is why we avoid a large one after flattening.

### Softmax
- **In one line:** Turns the 6 raw output scores into 6 probabilities that add up to 1.
- **Explained simply:** The output layer gives one score per class, and the scores can be any size. Softmax makes every score positive and rescales them so the total is 1. The biggest score gets the biggest share. The predicted class is the one with the highest probability.
- **Formula (if any):** p_i = e^(z_i) / (sum over j of e^(z_j)), where z are the raw scores.
- **Why we used it here:** We need a probability for each of the 6 defect types, and cross-entropy loss expects probabilities.
- **Where in our code:** Notebook 04, section 'M1', and notebook 05, section 'M3' (`activation="softmax"` on the last dense layer).
- **Common confusion / trap:** A probability of 0.9 does not mean the model is right 90 % of the time. It only shows how confident the model is.

### Cross-entropy loss
- **In one line:** A number that is small when the model gives a high probability to the true class and large when it does not.
- **Explained simply:** The loss is the model's penalty for one image. If the true class is scratches and the model says 0.95 for scratches, the penalty is small. If it says 0.05, the penalty is large. Training changes the weights to make the average penalty smaller. We call it "sparse" because our labels are class numbers (0 to 5), not one-hot vectors.
- **Formula (if any):** Loss for one image = minus the natural log of the probability given to the true class.
- **Why we used it here:** It is the standard loss for multi-class classification. We also use validation loss to pick the best epoch and to break the M1 and M3 tie.
- **Where in our code:** Notebooks 04 and 05, section 'Training setup shared by all models' (`sparse_categorical_crossentropy`).
- **Common confusion / trap:** Loss and accuracy are different. A model can keep the same accuracy while its loss falls, because it becomes more confident on images it already gets right. M3's validation loss (0.0832) is lower than M1's (0.1056) at the same 97.04 % accuracy.

### Adam and learning rate
- **In one line:** Adam is the rule that updates the weights after each batch. The learning rate is the size of each update.
- **Explained simply:** After each batch the network knows which way to nudge every weight to lower the loss. The learning rate says how big that nudge is. Too large and the loss jumps around. Too small and training is slow. Adam adjusts the step size for each weight separately, using the recent history of its gradients, so we have to tune less by hand.
- **Formula (if any):** New weight = old weight minus (learning rate × an adjusted gradient).
- **Why we used it here:** Adam works well with little tuning. We start at 1e-3 (0.001) for M1, M2 and MobileNetV2 stage A, and use 1e-4 for stage B so the pretrained weights are only adjusted gently. ReduceLROnPlateau halves it when learning stalls.
- **Where in our code:** Notebooks 04 and 05, `train()` in the section 'Training setup shared by all models' (`keras.optimizers.Adam(lr)`).
- **Common confusion / trap:** Adam does not remove the need for a sensible learning rate. 1e-3 is a good start for a new network, but too big for fine-tuning a pretrained one.


## Training & regularisation

### Epoch, batch and step
- **In one line:** An epoch is one pass over all training images, a batch is a group of images, a step is one weight update.
- **Explained simply:** We have 1,260 training images and a batch size of 32. The network processes one batch, computes the loss and updates the weights once. That is one step. After 40 steps it has seen every training image once, which is one epoch. Training runs for many epochs. M1 used all 100 epochs we allowed.
- **Formula (if any):** Steps per epoch = ceil(1260 / 32) = 40.
- **Why we used it here:** We watch the loss and accuracy epoch by epoch to see when the model stops improving. Early stopping and ReduceLROnPlateau also count in epochs.
- **Where in our code:** Notebooks 04 and 05, `train()` (`model.fit(..., epochs=epochs)`) and `BATCH = 32` in the pipeline section.
- **Common confusion / trap:** One epoch is not one update. It is 40 of them. See also the entry 'Batch' above.

### Overfitting
- **In one line:** The model does very well on the training images but worse on images it has not seen.
- **Explained simply:** A model with enough weights can memorise its training images, including their noise, instead of learning the real patterns. The sign is a training curve that keeps improving while the validation curve stops or gets worse. We keep the validation images away from training for this reason.
- **Formula (if any):** None. The gap between training accuracy and validation accuracy measures it.
- **Why we used it here:** With 210 training images per class, it is the main risk. We used augmentation, dropout, global average pooling, early stopping and transfer learning against it. In M1 it did not happen: its training accuracy (dropout on) was below its validation accuracy in 91 of 100 epochs.
- **Where in our code:** Notebook 04, sections 'M1' and 'What we found' (curves from `plot_curves`).
- **Common confusion / trap:** Training accuracy below validation accuracy is not a bug. Dropout is on while training and off while validating, so the training number is measured on a handicapped network.

### Dropout
- **In one line:** During training, switch off a random part of the units so the network cannot lean on any one of them.
- **Explained simply:** At each step, every unit of the layer is turned off with a given probability. The rest must still get the answer right, so the units learn to be useful on their own instead of relying on a few partners. At prediction time dropout is off and all units work.
- **Formula (if any):** None. Rate 0.5 means half the units are off in each step.
- **Why we used it here:** It is cheap protection against overfitting. M1 and M2 use rate 0.5 after the 128-unit dense layer. MobileNetV2's head uses 0.3.
- **Where in our code:** Notebook 04, section 'M1' (`Dropout(0.5)`); notebook 05, section 'M3' (`Dropout(0.3)`).
- **Common confusion / trap:** Dropout acts only while training. That is one reason the training accuracy in the log can be lower than the validation accuracy.

### Batch normalisation (BN)
- **In one line:** A layer that rescales its inputs to a steady range, using the batch in training and saved averages in prediction.
- **Explained simply:** Inside a network the numbers can drift to odd sizes from layer to layer. BN fixes them: it subtracts the mean and divides by the spread, then lets the network learn a scale and a shift. This makes training steadier. **In training**, BN uses the mean and variance of the current batch of 32 images. **In prediction**, there may be a single image, so BN cannot use a batch. It uses running averages of the mean and variance, collected during training. After every step the running average moves a little toward the new batch value. The momentum says how much of the old value is kept: new average = momentum × old average + (1 minus momentum) × batch value.
- **Formula (if any):** Output = scale × (x minus mean) / sqrt(variance + a tiny number) + shift.
- **Why we used it here:** M2 has BN after each convolution. It also caused both of our bugs.
  - **Bug 1, M2 (Phase 4.1).** The Keras default momentum is 0.99, so each step moves the average only 1 %. With 40 steps per epoch, the averages lagged far behind. In the first BN layer the running variance was about 0.67, while the real variance was about 0.01. Every validation image was normalised wrongly and M2 scored 16.67 % (one class for all images). Momentum 0.9 fixed it (95.19 %, then 95.93 % in the final run).
  - **Bug 2, MobileNetV2 (Phase 4.2).** When we unfroze the base for fine-tuning, its BN layers began to use the statistics of each steel batch instead of the ImageNet ones. Training accuracy fell from 0.90 to 0.51 in the first fine-tuning epoch, although the learning rate was tiny. Keeping all BN layers frozen fixed it (88.89 % to 97.41 %).
- **Where in our code:** Notebook 04, section 'M2' (`BatchNormalization(momentum=0.9)`); notebook 05, section 'M3', stage B cell (`not isinstance(layer, keras.layers.BatchNormalization)`).
- **Common confusion / trap:** A model whose training curve looks fine but scores exactly 1 in 6 on validation points to BN. The cause is the difference between training and prediction behaviour, not the data. BN also has its own parameters: M2 has 110,630 against 109,958 for M1, because BN adds a scale, a shift and two running averages per channel.

### Early stopping (and patience)
- **In one line:** Stop training when the validation loss has not improved for a while, and go back to the best epoch.
- **Explained simply:** After the best epoch the model usually starts to memorise the training images, and the validation loss gets worse. Patience is how many epochs we wait without improvement before stopping. We then restore the weights from the best epoch, not the last one.
- **Formula (if any):** None.
- **Why we used it here:** It saves time and limits overfitting. M1 and M2 use patience 12. We first used 8, but M2's validation curve is jumpy and training stopped before the learning-rate cuts could calm it. MobileNetV2 uses 5 in stage A and 8 in stage B. M2 stopped after 41 epochs with its best at epoch 29. M1 never triggered it and ran all 100.
- **Where in our code:** Notebooks 04 and 05, `train()` (`EarlyStopping(monitor="val_loss", patience=patience, restore_best_weights=True)`).
- **Common confusion / trap:** A patience that is too small stops training during a normal bumpy patch. With only 270 validation images the curve is noisy.

### ReduceLROnPlateau
- **In one line:** Halve the learning rate when the validation loss stops improving.
- **Explained simply:** A big step is useful at the start, but near the best weights it makes the model jump around the minimum. When the validation loss has not improved for 4 epochs, we cut the learning rate by half (factor 0.5), so the model can settle into a better spot.
- **Formula (if any):** New learning rate = old learning rate × 0.5, never below 1e-7.
- **Why we used it here:** It calms the jumpy validation curve of M2 and helps the other models settle. Its patience (4) is shorter than early stopping's (12), so the learning rate can fall a few times before training is stopped.
- **Where in our code:** Notebooks 04 and 05, `train()` (`ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=4, min_lr=1e-7)`).
- **Common confusion / trap:** It changes the learning rate, not the weights directly. It is separate from early stopping, which ends training.

### Model checkpoint
- **In one line:** Save the model to a file whenever the validation loss reaches a new low.
- **Explained simply:** During a run, a callback checks the validation loss after every epoch. If it is the best so far, it writes the weights to a `.keras` file. If training later gets worse, the file still holds the best version.
- **Formula (if any):** None.
- **Why we used it here:** We want the best epoch, not the last. M1's file is 1.38 MB, M2's is 1.41 MB and M3's is 24.46 MB.
- **Where in our code:** Notebooks 04 and 05, `train()` (`ModelCheckpoint(..., monitor="val_loss", save_best_only=True)`).
- **Common confusion / trap:** The checkpoint picks by validation loss, not accuracy, and the two can disagree on the best epoch. In notebook 05, `record()` also saves the model again after we choose between the stage A and stage B weights.

### Transfer learning
- **In one line:** Start from a network already trained on a large dataset and adapt it to our task.
- **Explained simply:** A network trained on ImageNet (about 1.28 million training photos in 1,000 classes) has already learned to detect edges, corners and textures in its early layers. These help on steel too. We keep those layers and train a new final part for our 6 defect classes. This needs far fewer images than training from nothing.
- **Formula (if any):** None.
- **Why we used it here:** We have only 1,260 training images. Transfer learning gives M3, which scores 97.04 % and has the lowest validation loss (0.0832).
- **Where in our code:** Notebook 05, section 'M3' (`MobileNetV2(weights="imagenet", include_top=False)`).
- **Common confusion / trap:** It is not automatic. With the base frozen, M3 reached only 81.11 %, because steel textures look little like ImageNet photos. The adaptation step (fine-tuning) mattered.

### Fine-tuning (stage A / stage B)
- **In one line:** After training the new head, unfreeze part of the pretrained base and train it gently too.
- **Explained simply:** We do it in two stages. **Stage A:** the base is frozen (its weights cannot change). Only the new head learns, with learning rate 1e-3. Without stage A, the random head would send large wrong gradients into the pretrained layers. **Stage B:** we unfreeze the base from layer 100 on (its last six blocks, about a third of its layers) and keep training with a 10 times smaller learning rate, 1e-4, so the pretrained weights are nudged and not overwritten. All BN layers stay frozen. Stage A ran 15 epochs and stage B 14, and the saved weights come from epoch 21.
- **Formula (if any):** None.
- **Why we used it here:** Stage A alone peaked at 81.11 %. Stage B lifted M3 to 97.04 %. The learning rate in stage B is lower because the weights are already good and we only want small corrections.
- **Where in our code:** Notebook 05, section 'M3', the stage A cell and the stage B cell (`FINE_TUNE_FROM = 100`).
- **Common confusion / trap:** In Keras 3, calling the base with `training=False` is not enough to keep its BN layers fixed once they are unfrozen. Freeze them layer by layer. Stage B is also not exactly reproducible: the same settings gave 97.41 % in a check run and 97.04 % in the final run.

### MobileNetV2 (depthwise-separable convolutions, ImageNet)
- **In one line:** A small, fast CNN made for phones and edge devices, pretrained on ImageNet.
- **Explained simply:** A normal 3 × 3 convolution mixes all input channels at once, which takes many multiplications. MobileNetV2 splits the job in two. A depthwise step runs one 3 × 3 filter on each channel alone (it looks at space). Then a 1 × 1 convolution mixes the channels (it looks across channels). The result is close to a normal convolution, with several times fewer multiplications and weights. ImageNet (the ILSVRC-2012 set) has about 1.28 million labelled training photos in 1,000 classes, used to pretrain it.
- **Formula (if any):** None. Our M3 has 2,265,670 parameters, about 20 times more than M1 (109,958).
- **Why we used it here:** It is small enough for a real-time line or an edge device, and it has well-tested pretrained weights. Its input must be 224 × 224 with 3 channels, so we copy our grey image into 3 channels and rescale the pixels with `preprocess_input`.
- **Where in our code:** Notebook 05, sections 'Pipeline' (`three_channels=True`, 224 pixels) and 'M3'.
- **Common confusion / trap:** "Mobile" describes the design, not our file: M3's file is 24.46 MB, about 18 times the size of M1 (1.38 MB). It is light compared with big CNNs, not compared with our tiny one.


## Evaluation & explainability

### Accuracy
- **In one line:** The share of images the model labels correctly.
- **Explained simply:** Count the images where the predicted class equals the true class, then divide by the number of images. If 266 of 270 test images are right, accuracy is 266 / 270. It is easy to read, and it is fair when every class has the same number of images, as in NEU.
- **Formula (if any):** Accuracy = correct predictions / all predictions.
- **Why we used it here:** It is our headline number. M3 scored 98.52 % on the test set (4 errors in 270). One test image is worth 0.37 percentage points.
- **Where in our code:** Notebook 06, section 'Predictions and metrics'.
- **Common confusion / trap:** On an unbalanced dataset accuracy can look high while a rare class is always missed. Our data is balanced, but we still look at per-class numbers. Also, a gap of one or two images between models is noise, not a real difference.

### Precision, recall and F1
- **In one line:** Precision asks "when the model says X, is it right?". Recall asks "of all real X, how many did it find?". F1 combines the two.
- **Explained simply:** Take the crack class. Precision = of the images the model called crack, how many really were cracks. Recall = of the real crack images, how many the model found. A model that calls everything a crack has perfect recall and poor precision. A model that calls almost nothing a crack has the opposite. F1 is the harmonic mean of the two, so it is high only when both are high.
- **Formula (if any):** Precision = TP / (TP + FP). Recall = TP / (TP + FN). F1 = 2 x precision x recall / (precision + recall). TP is a true positive, FP a false alarm, FN a miss.
- **Why we used it here:** A missed crack ships to the customer, so recall matters most for cracks. A false alarm only costs a quick re-check. On the test set all three models had crack recall 1.0000. Crack precision was 1.0000 for M3, 0.9783 for M1 and 0.9375 for M2.
- **Where in our code:** Notebook 06, section 'Predictions and metrics' (scikit-learn classification reports).
- **Common confusion / trap:** The two are not the same. High precision with low recall means careful but misses a lot. The harmonic mean is pulled down by the smaller value, which is the point.

### Macro average
- **In one line:** Work out a metric for each class, then take the plain average of the six numbers.
- **Explained simply:** Every class gets the same weight, whether it is easy or hard. A weak class cannot hide behind strong ones. Macro F1 is the average of the six per-class F1 scores.
- **Formula (if any):** Macro F1 = (F1 of class 1 + ... + F1 of class 6) / 6.
- **Why we used it here:** All classes matter equally to the plant. M3's macro F1 is 0.9852 and M1's is 0.9400. With 45 test images in every class, macro F1 sits very close to accuracy.
- **Where in our code:** Notebook 06, section 'Predictions and metrics'.
- **Common confusion / trap:** Macro is not the same as weighted or micro averaging. Those weigh classes by size. Because NEU is balanced, the choice changes little here.

### Crack-vs-rest metrics (one-vs-rest)
- **In one line:** Turn the 6-class problem into a yes or no question: crazing or not crazing.
- **Explained simply:** We call crazing "positive" and the other five classes together "negative". Then we compute precision, recall and F1 for the positive class. The problem statement asks about cracks, so this gives a direct answer. In the test set, 45 images are crazing and 225 are not.
- **Formula (if any):** Same as precision and recall, with TP = crazing images called crazing, FN = crazing images called anything else, FP = other images called crazing.
- **Why we used it here:** It answers "can you find cracks?" in one line. All three models found all 45 crazing images (recall 100 %). M1 called one patches image crazing (precision 0.9783) and M2 called three other images crazing (precision 0.9375).
- **Where in our code:** Notebook 06, section 'Predictions and metrics'.
- **Common confusion / trap:** The model still predicts six classes. We only change how we score it. A wrong guess between two non-crack classes does not count here at all.

### Confusion matrix (counts vs row-normalised)
- **In one line:** A table that shows, for each true class, which classes the model predicted.
- **Explained simply:** Rows are the true class and columns are the predicted class. The diagonal holds the correct answers. Any number off the diagonal is a mix-up, and its position tells you which two classes got confused. The counts version shows how many images. The row-normalised version divides each row by its total, so each cell is a share of that class, and a weak class stands out.
- **Formula (if any):** Row-normalised cell = count / (number of images in that true class). The diagonal of the normalised matrix is the recall of each class.
- **Why we used it here:** It shows which defects get mixed up, which one accuracy number cannot. M3's four errors are three scratches called inclusion and one rolled-in scale called pitted surface. M1's weak class is pitted surface (recall 77.78 %).
- **Where in our code:** Notebook 06, section 'Confusion matrices of the finalists' (figures `eval_01` and `eval_02`).
- **Common confusion / trap:** Do not mix up rows and columns. Reading across a row gives recall. Reading down a column gives precision.

### Inference latency (CPU, one image at a time)
- **In one line:** The time the model needs to classify one image.
- **Explained simply:** We time 100 single-image calls and take the mean, after 10 warm-up calls that are not counted. The first calls are slow because the library is still setting things up. We use the CPU because a small computer next to the line may have no GPU. We use one image at a time because on a line each image must be answered as it arrives, not saved up into a batch.
- **Formula (if any):** Latency = total time of 100 calls / 100.
- **Why we used it here:** A strip moves fast, so speed limits what the system can do. On Kaggle's CPU, M1 takes 21.57 ms per image and M3 takes 223.96 ms, about 10 times slower. After TensorFlow Lite conversion M3 takes 18.03 ms.
- **Where in our code:** Notebook 06, section 'Speed and size'. Notebook 07, section 'I4 · TensorFlow Lite for a small device'.
- **Common confusion / trap:** The numbers belong to Kaggle's CPU. A laptop or an edge board will give different times. Compare models on the same machine, not across machines.

### Grad-CAM
- **In one line:** A heatmap that shows which parts of the image pushed the model towards its answer.
- **Explained simply:** We pick the last convolution layer, because it still keeps some position information and already holds high-level patterns. Then, step by step:
  1. Run the image and take the score of the predicted class.
  2. Compute the gradient of that score with respect to every feature map of the last convolution layer. The gradient tells how much the score would change if a spot in the map changed.
  3. Average each map's gradient over the whole map. This gives one weight per map: how important that pattern is for this class.
  4. Multiply each feature map by its weight and add the maps together.
  5. Apply ReLU, so we keep only the places that push the score up.
  6. Scale the small map up to the image size and lay it over the image as a colour overlay. Red means a strong influence.
- **Formula (if any):** weight_k = average over positions of (d score / d map_k). Heatmap = ReLU(sum over k of weight_k x map_k).
- **Why we used it here:** To check that the CNN looks at the defect and not at something else. M3's heatmaps sit on the scratch line, the inclusion streak and the dark patches. M1's often sit on the flat background.
- **Where in our code:** Notebook 07, section 'I1 · Grad-CAM: where does the model look?' (written by hand with `tf.GradientTape`). For M1 the layer is the last ReLU before pooling. For M3 it is the output of the pretrained base (7 x 7 cells, 1,280 channels).
- **Common confusion / trap:** The heatmap is coarse. For M3 it starts as only 7 x 7 cells, so it shows the area, not the exact outline. It shows where the model looked, not proof that its answer is right.

### Shortcut learning
- **In one line:** The model solves the task with an easy clue that does not truly mark the defect.
- **Explained simply:** A network learns whatever lowers the loss fastest. If a clue such as overall brightness or a flat background is enough on the training data, it can use that and ignore the defect. On images that look like the training set, the shortcut works. When the images change, it fails. A student who learns last year's answers instead of the subject is the same case.
- **Formula (if any):** None.
- **Why we used it here:** It is our explanation for M1. Inclusion images are the flattest in our EDA, and M1's heatmaps for inclusion light up the flat steel around the defect. When we darkened the validation images by 0.15, M1 fell from 97.0 % to 65.9 %, and by 0.30 to 31.9 %. M3 stayed at 96.7 % for both. This also fits M1's weak pitted-surface class on the test set. We treat it as a strong hint, not as proof.
- **Where in our code:** Notebook 07, sections 'I1 · Grad-CAM: where does the model look?' and 'I2 · Robustness: dirty lens, blur and bad lighting'.
- **Common confusion / trap:** A high test score does not rule out a shortcut, because the test set comes from the same source as the training set. Heatmaps and robustness tests are the way to find it.

### Robustness testing (corruption types and severities)
- **In one line:** Damage the images in controlled ways and measure how much accuracy falls.
- **Explained simply:** A real plant has dirty lenses, shaking, bad lighting and sensor noise. NEU has none of that. So we add each kind of damage at three severities and score the models again. We used four corruption types: Gaussian noise (standard deviation 0.02, 0.05, 0.10), blur (kernel 3, 5, 9 px), darker images (0.15, 0.30, 0.45 subtracted) and lower contrast (multiplied by 0.8, 0.6, 0.4). A steep fall means the model depends on fragile details.
- **Formula (if any):** None. Darker = pixel minus the amount, on the 0 to 1 scale.
- **Why we used it here:** It tests the claim that 98.5 % is too good to be true. At noise 0.10, validation accuracy falls to 24.4 % for M1, 25.2 % for M2 and 48.1 % for M3. M3 is the most robust: 92.6 % with a 5 px blur and 91.9 % with images 0.45 darker.
- **Where in our code:** Notebook 07, section 'I2 · Robustness: dirty lens, blur and bad lighting'. It uses the validation set, so the test set stays single-use.
- **Common confusion / trap:** These are made-up corruptions, not real plant images. They show weak spots. They do not predict the exact accuracy on a real line.

### Ablation study
- **In one line:** Remove one part of the recipe, retrain, and see what changes.
- **Explained simply:** To learn what a part is worth, we run the same experiment with and without it, keeping everything else equal (same network, seed and callbacks). We took the M2 network and trained it three ways: no augmentation, augmentation without rotation, and the full set. We compared them on the validation set.
- **Formula (if any):** None.
- **Why we used it here:** On clean validation images, accuracy was 100.00 % with no augmentation, 99.26 % without rotation and 95.56 % with full augmentation. So augmentation costs accuracy on clean NEU images, and rotation costs the most. Scratches and rolled-in scale follow the rolling direction, so a tilt is not always a harmless change. But augmentation is what keeps M2 working when the light changes (89.6 % at darker by 0.15, against 65.9 % for M1).
- **Where in our code:** Notebook 07, section 'I3 · Augmentation ablation'.
- **Common confusion / trap:** Training on the GPU is not exactly repeatable. An earlier run gave the same order (100.00, 98.89, 94.07 %) but different numbers. Read the order, not the decimals.

### TensorFlow Lite and dynamic-range quantisation
- **In one line:** A compact model format for small devices, with weights stored as 8-bit integers.
- **Explained simply:** TensorFlow Lite converts a Keras model into a small file that runs on phones and boards. Quantisation reduces the number of bits per weight. A normal weight is a 32-bit float. Dynamic-range quantisation stores each weight as an 8-bit integer plus a scale factor that maps it back, so the weights take about four times less space. It needs no calibration images, which is why we chose it over full integer quantisation. The saved file shrinks even more than four times, because a `.keras` file also stores the optimiser's state (Adam keeps two extra numbers per weight): M1's 109,958 weights take 0.44 MB as 32-bit floats but its `.keras` file is 1.38 MB, and its TFLite file is 0.12 MB, about one byte per weight.
- **Formula (if any):** Real weight is about integer x scale. Size ratio of the weights = 8 / 32 = 1/4.
- **Why we used it here:** M3 shrank from 24.46 MB to 2.51 MB at 97.41 % validation accuracy, and gave the same answer as Keras on 99.63 % of images. M1 shrank from 1.38 MB to 0.12 MB at 97.04 %, with 100 % the same answers.
- **Where in our code:** Notebook 07, section 'I4 · TensorFlow Lite for a small device'.
- **Common confusion / trap:** Our first conversion of M3 failed, because its random augmentation layers use operations that TensorFlow Lite cannot convert. We convert an inference copy without the augmentation block, which does nothing at prediction time anyway. Quantisation can change answers slightly, so always check agreement with the original.
