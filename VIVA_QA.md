# VIVA Q&A: EST Project (Steel Surface Defect CNN)

Grouped by phase. Add questions as you go. Every number comes from `outputs/results/` or `LOG.md`.
**Mock viva:** pair up and quiz each other with 10 random questions from `VIVA_QA.md`, one at a time, grading against the answers.

## Problem & motivation

**Q:** Why automate steel surface inspection?
**A:** The strip moves very fast, and human inspectors get tired and judge inconsistently. A camera + CNN checks every metre the same way, 24×7.

**Q:** The statement says "cracks". Where are the cracks in your project?
**A:** NEU's crazing class is a network of fine surface cracks. We classify all 6 defect types and also report crack-specific metrics: on the test set our final model (MobileNetV2) found all 45 crazing images (crack recall 100 %) with no false crack alarms (crack precision 100 %).

**Q:** How is this "predictive maintenance"?
**A:** Trends in defect types point to equipment problems. For example, rising rolled-in scale suggests checking the descaling system, and rising scratches suggest checking the rolls and guides. So maintenance can be scheduled before quality drops.

**Q:** For cracks, which matters more: precision or recall?
**A:** Recall. A missed crack (false negative) ships to the customer. A false alarm only costs a quick re-check.

**Q:** How is hot-rolled steel strip made, in short?
**A:** A slab is heated to roughly 1100 to 1250 degrees C, descaled with high-pressure water, rolled thinner in a roughing mill and a finishing mill, cooled with water sprays and coiled.

**Q:** Why do surface defects matter?
**A:** They make the product look worse, can lower its fatigue resistance, and cause customer complaints and cost. Catching them at the mill is better than at the customer.

**Q:** Which of your six defect types points to which equipment?
**A:** Rolled-in scale points to the descaling system (low pressure or blocked nozzles), and scratches point to the rolls and roll gap. Inclusions come from steelmaking and casting, so they point upstream of the rolling mill. A trend is a hint to inspect, not proof, because one defect can have several causes.

**Q:** What is crazing, and what causes it?
**A:** A network of fine surface cracks, and it is our crack class. We did not find a source that explains its cause, so we do not claim one.

**Q:** Which two classes look most alike, and why does that matter?
**A:** Crazing and rolled-in scale share visual similarities in the literature. We check the confusion matrix for mix-ups between them.

**Q:** What is a line-scan camera?
**A:** A camera with one row of pixels that builds a long image as the strip moves past. A real line therefore gives long, high-resolution images, unlike NEU's 200 x 200 crops.

**Q:** NEU images are 200 x 200 but the statement says high-resolution images. How do you handle that?
**A:** NEU is a benchmark of small crops. A real system would cut a long image into tiles and classify each tile. We explain this, and demonstrate it only if time allows.

**Q:** What does "automated visual inspection" add over a human inspector?
**A:** It applies the same rule to every image and does not tire. Human inspection is slow, error-prone and depends on experience.

## Data

**Q:** What dataset did you use?
**A:** The NEU surface defect database: 1,800 grayscale 200×200 images of hot-rolled steel, 300 for each of 6 defect types.

**Q:** Why split before augmentation?
**A:** If augmented copies of one image ended up in both train and test, the model would be tested on near-copies of what it trained on, which gives fake accuracy.

**Q:** Why is the test set never augmented?
**A:** The test set must look like real, unseen images from the line. Augmentation is only a training trick.

## CNN basics

**Q:** What is a CNN?
**A:** A neural network that slides small learnable filters over the image to detect local patterns (edges, textures), then combines them into bigger patterns to recognise the class.

**Q:** What does a convolution layer do?
**A:** It slides small filters (e.g. 3×3) across the image. Each filter produces a feature map showing where its pattern appears.

**Q:** Why use pooling?
**A:** It shrinks the feature maps, which means fewer computations and parameters, and it makes the network less sensitive to the exact position of a pattern.

**Q:** What is ReLU?
**A:** The activation function max(0, x). It adds non-linearity so the network can learn complex patterns, and it trains fast.

**Q:** What does softmax do?
**A:** It turns the final 6 scores into probabilities that add up to 1.

**Q:** Which loss did you use?
**A:** Sparse categorical cross-entropy. It is large when the model gives a low probability to the true class.

## Training

**Q:** What is overfitting, and how did you prevent it?
**A:** The model memorises the training images and fails on new ones. We used augmentation, dropout, BatchNorm, early stopping and transfer learning. The train vs validation curves show the effect.

**Q:** What is transfer learning?
**A:** We reuse a network already trained on ImageNet (about 1.28 million training photos, 1,000 classes), whose early layers already detect edges and textures, and retrain only the top part for our 6 defect classes. It needs much less data.

**Q:** Why MobileNetV2?
**A:** It's small and fast (it uses depthwise-separable convolutions), so it suits a real-time production line or a small edge device.

## Evaluation & innovation

**Q:** Your accuracy is very high. Isn't that suspicious?
**A:** NEU images are clean, balanced and taken in controlled conditions. We checked for duplicates and leakage, and our robustness test shows the limits: with strong noise (std 0.10) MobileNetV2's validation accuracy drops from 97.0 % to 48.1 %, and with a 9 px blur to 84.8 %. A real plant would need its own data.

**Q:** What is Grad-CAM?
**A:** It uses the gradients of the predicted class with respect to the last convolution layer to build a heatmap of the image regions that drove the decision. It shows our CNN looks at the defect itself.

## Data check

**Q:** How many images are there, and what size and colour mode are they?
**A:** 1,800 images, 300 in each of the 6 classes. All are 200 x 200 px JPEG files, stored as RGB.

**Q:** NEU is called grayscale, but the files are RGB. Which is it?
**A:** Both. All 1,800 files are stored in RGB format, but in every one the three channels are identical, so they are grayscale pictures saved in a colour file. Our own CNN reads one channel. For MobileNetV2 we repeat that channel three times.

**Q:** Did you check for duplicate images? What did you find?
**A:** Yes. We hashed every file (MD5 of the file bytes) and every decoded picture (MD5 of the pixels). One pair is identical: `patches_101.jpg` and `patches_105.jpg`, both labelled patches. Both hashes found the same pair.

**Q:** What is an MD5 hash and why does it find duplicates?
**A:** It turns the data into a 32-character fingerprint. Identical data always gives the same fingerprint, and any change gives a different one. So two images with the same fingerprint are copies.

**Q:** The Kaggle mirror already has train and validation folders. Why did you not use them?
**A:** They hold 1,440 and 360 images and have no test set. We made our own 70/15/15 split so that we control it, can stratify it, and keep identical pictures together.

## Split & EDA

**Q:** How did you split the data?
**A:** Seed 42, stratified 70/15/15, in groups of identical pixels. That gives 1,260 train, 270 validation and 270 test images, which is 210 / 45 / 45 per class.

**Q:** What does stratified mean, and why use it?
**A:** Each class keeps the same share in every part. Without it, a class could be rare in the test set by bad luck, and the test score would be misleading.

**Q:** Why split by groups of identical pixels instead of single images?
**A:** Because of the duplicate pair. If one copy were in training and the other in test, the test set would contain an image the model had already seen. Keeping each group in one part prevents that. We checked: 0 pictures are in two parts.

**Q:** What is data leakage? How did you avoid it?
**A:** Leakage means test information reaches training, so the test score looks too good. We split before exploring or augmenting, kept identical pictures together, never augment validation or test images, and did all EDA on training images only.

**Q:** Why do the EDA only on the training images?
**A:** If we studied the test images, our design choices could quietly fit them, and the test would no longer be a fair surprise.

**Q:** What did the brightness and contrast analysis show?
**A:** The median brightness by class runs from 90.36 (scratches) to 186.05 (pitted surface), but the ranges overlap across classes. Patches have by far the highest median contrast (57.75; the others are between 11.18 and 28.55). So brightness alone cannot separate the classes.

**Q:** What did the pixel histograms show?
**A:** The class histograms overlap heavily. Patches and pitted surface have a spike at pixel value 255, which means over-exposed parts of the image.

**Q:** What did the mean images show?
**A:** Crazing, inclusion, pitted surface and scratches darken towards the right edge, which points to uneven lighting from the camera setup. The scratches mean shows both vertical and horizontal lines. A model could learn the lighting instead of the defect, which is one reason we use augmentation with brightness and contrast changes.

**Q:** What did PCA tell you?
**A:** On 50 x 50 downscaled pixels, PC1 holds 70.63 % of the variance and PC2 holds 8.28 %. Pitted surface sits to the right, scratches to the left, and crazing, inclusion and rolled-in scale overlap in the middle. So raw pixels do not separate all six classes, and we need a model that learns features.

**Q:** Is the dataset balanced? Why does that matter?
**A:** Yes, 300 images per class, and 210 / 45 / 45 per class after the split. Plain accuracy is therefore a fair first metric and no class needs extra weight.

## Pre-processing & augmentation

**Q:** What input does each model get?
**A:** The scratch CNN gets 200 x 200 x 1 images scaled to 0 to 1. MobileNetV2 gets 224 x 224 x 3 images (the grayscale channel repeated) scaled to -1 to 1 by `preprocess_input`.

**Q:** Why does MobileNetV2 need `preprocess_input`?
**A:** It was pretrained on images scaled to -1 to 1. With a different scale nothing crashes, but it trains badly.

**Q:** Which augmentations did you use, and why those?
**A:** Horizontal and vertical flip, rotation of 0.05 of a full turn (about plus or minus 18 degrees) with reflect fill, zoom 0.1, contrast 0.2 and brightness 0.2. Each matches something that really happens on the line: a tilt, a change of distance, changing light. We avoid 90 degree turns, because scratches and rolled-in scale follow the rolling direction.

**Q:** Why augment at all?
**A:** We have only 210 training images per class, so a CNN could memorise them. Augmentation shows a slightly different version of each image every epoch, which helps against overfitting.

**Q:** Why is augmentation applied only to training images?
**A:** Validation and test images stand in for real, unseen images, so they must stay unchanged. An augmented test set would not measure what happens on the line.

**Q:** Why are the augmentation layers inside the model?
**A:** Keras turns random layers on only in training mode. Validation, test and prediction skip them automatically, so we cannot forget to switch them off. It also runs on the GPU with the model.

**Q:** Why `fill_mode="reflect"`?
**A:** A rotation or zoom exposes empty corners. Reflect fills them with a mirror image of the steel surface instead of black triangles, which never appear on a real strip.

**Q:** What is a batch, and what batch size do you use?
**A:** A batch is a group of images processed together in one training step. We use 32. The weights are updated once per batch, and one epoch is about 40 batches (1,260 / 32).

**Q:** What do `prefetch` and `cache` do in a `tf.data` pipeline?
**A:** `prefetch` prepares the next batch while the GPU works on the current one. `cache` keeps decoded images in memory after the first epoch. Notebook 03 uses `prefetch` only; notebook 04 (training) adds `cache` and shuffles after it, so the order still changes every epoch.

## Models & training

**Q:** Explain your M1 network layer by layer.
**A:** The input is a 200 × 200 grey image. Three blocks of 3 × 3 convolution, ReLU and 2 × 2 max pooling, with 32, 64 and 128 filters, shrink the image 200, 100, 50, 25 pixels. Global average pooling turns the 128 maps into 128 numbers. A dense layer of 128 units, dropout 0.5 and a dense layer of 6 units with softmax follow. In total: 109,958 parameters.

**Q:** Why global average pooling instead of flatten?
**A:** Flatten would give 25 × 25 × 128 = 80,000 numbers, and a dense layer on top of them could memorise our 1,260 images. GAP gives 128 numbers and adds no weights. It is a main reason M1 has only 109,958 parameters and does not overfit.

**Q:** Did M1 overfit?
**A:** No. Its training accuracy (measured with dropout on) was below its validation accuracy in 91 of 100 epochs. It reached 97.04 % validation accuracy and was still improving slowly at the 100-epoch limit.

**Q:** Why did M2 (batch norm and augmentation) score lower than M1?
**A:** M2 got 95.93 % and M1 97.04 %, which is three validation images. M1 is tiny and already has dropout and GAP, so there was little overfitting for M2 to fix. M2's validation curve is also jumpy (72.6 % to 96.3 % after epoch 10), so a three-image gap is within the noise.

**Q:** Why did M2 first score 16.7 % on validation?
**A:** 16.67 % is exactly 1 in 6, so the model gave one class for every image. Batch norm predicts with running averages, and the Keras default momentum of 0.99 moves them only 1 % per step. With 40 steps per epoch they lagged far behind (running variance about 0.67, real value about 0.01), so every validation image was normalised wrongly.

**Q:** What is batch norm momentum, and what did you change?
**A:** It sets how fast the running average follows the batch values: new average = momentum × old + (1 minus momentum) × batch value. At 0.99 it is too slow for 40 steps per epoch. We set 0.9, which catches up within one epoch, and M2 then scored 95.19 % in the repair run.

**Q:** Why does batch norm behave differently in training and prediction?
**A:** In training it normalises with the mean and variance of the current batch of 32. In prediction there may be one image only, so it uses running averages saved during training. If those averages are wrong, training looks fine but validation fails. That is what happened to M2.

**Q:** Why did MobileNetV2 first score only 81 to 89 %, and how did freezing batch norm fix it?
**A:** Frozen features alone gave 81.11 %, and our first fine-tuning try gave 88.89 %. In fine-tuning the unfrozen batch norm layers switched to the statistics of steel batches and damaged the ImageNet features: training accuracy fell from 0.90 to 0.51 in the first epoch. With every batch norm layer kept frozen we got 97.41 % (97.04 % in the final run).

**Q:** Why do you train MobileNetV2 in two stages?
**A:** In stage A only the new head learns on top of the frozen base, at learning rate 1e-3. Otherwise the random new head would send large wrong updates into the pretrained layers. Stage B then unfreezes the base from layer 100 on and fine-tunes it. Stage A alone reached 81.11 %, stage B 97.04 %.

**Q:** Why is the learning rate 1e-4 in stage B and 1e-3 in stage A?
**A:** The head starts from random numbers and needs big steps. The base already holds good weights, so we use a 10 times smaller rate to adjust them and not overwrite them. Our first try at 1e-5 learned too slowly (88.89 %).

**Q:** How did you choose the two finalists?
**A:** On validation data only, before opening the test set. M1 and M3 tie at 97.04 %. One image is 0.37 points, so we used validation loss as the tie-breaker: M3 has 0.0832 against 0.1056 for M1. We also kept M1 as the best of our own CNNs and about 18 times smaller in file size.

**Q:** How many parameters does each model have, and how big are the files?
**A:** M1 has 109,958 parameters (1.38 MB), M2 has 110,630 (1.41 MB) and M3 has 2,265,670 (24.46 MB). M2 has a few more than M1 because batch norm adds a scale, a shift and two running averages per channel.

**Q:** Why was the test set not used in this phase?
**A:** If we chose models by test results, the test score would no longer be an honest estimate for new images. So we pick models on the validation set and open the test set once, for the two finalists (M2 is scored too, only for the comparison chart).

**Q:** M1 and M3 differ by one or two validation images. Is one better?
**A:** We cannot say. The validation set has 270 images, so one image is 0.37 percentage points. A gap of one or two images is inside the noise. M3's re-run of the same settings gave 97.41 % once and 97.04 % once, which shows the same wobble.

**Q:** What are early stopping and patience, and what values do you use?
**A:** Early stopping stops training when validation loss has not improved for `patience` epochs, then restores the best epoch's weights. For M1 and M2 we use 12, raised from 8, because M2's jumpy curve stopped before the learning rate cuts could calm it. M2 stopped after 41 epochs with its best at epoch 29.

**Q:** What does ReduceLROnPlateau do?
**A:** It halves the learning rate (factor 0.5) when validation loss has not improved for 4 epochs. Smaller steps let the model settle near its best weights. It reacts before early stopping (12), so the learning rate can fall several times before training ends.

**Q:** What are an epoch, a batch and a step in your training?
**A:** The batch is 32 images and one step is one weight update. We have 1,260 training images, so one epoch is 40 steps. M1 trained for 100 epochs, which is 4,000 updates.

**Q:** Why did you use Adam, and with which learning rate?
**A:** Adam sets a step size for each weight from its recent gradients, so it needs little tuning. We start at 1e-3 for the models from scratch and for stage A, and use 1e-4 for stage B.

**Q:** What does a depthwise-separable convolution do, and why did you pick MobileNetV2?
**A:** It splits a normal convolution into one 3 × 3 filter per channel and then a 1 × 1 convolution that mixes the channels. That needs several times fewer multiplications for a similar result. It suits a real-time line or an edge device, and ImageNet weights are available.

**Q:** Why not just use MobileNetV2, since it has the lowest loss?
**A:** It ties M1 on validation accuracy (97.04 %) but is about 20 times bigger (2,265,670 against 109,958 parameters). So we carry both to the test and compare accuracy against size there.

## Evaluation

**Q:** What is your final test result?
**A:** We opened the test set (270 images) once. Our final model, M3 (MobileNetV2), got 98.52 % accuracy and macro F1 0.9852, with 4 mistakes. Our small CNN, M1, got 94.07 % with 16 mistakes. M2 got 96.67 % with 9.

**Q:** Why does crack recall matter most, and what is your value?
**A:** A missed crack (false negative) ships to the customer. A false alarm only costs a quick re-check. On the test set all three models found all 45 crazing images, so crack recall is 100 %. Crack precision is 1.0000 for M3, 0.9783 for M1 and 0.9375 for M2.

**Q:** What is the difference between precision, recall and F1?
**A:** Precision: of the images called crack, how many were cracks. Recall: of the real cracks, how many we found. F1 is the harmonic mean of the two, so it is high only when both are high.

**Q:** What does macro average mean, and why use it?
**A:** We compute the metric for each of the 6 classes and average the six numbers. Every class counts the same, so a weak class cannot hide. M3's macro F1 is 0.9852 and M1's is 0.9400.

**Q:** How do you read M1's confusion matrix?
**A:** Rows are true classes, columns are predicted classes. M1's weak class is pitted surface: recall 77.78 %. Of its 45 test images, 6 were called inclusion, 3 scratches and 1 rolled-in scale. Scratches recall is 88.89 % (4 called inclusion). 10 of M1's 16 mistakes are pitted surface.

**Q:** Which mistakes did M3 make?
**A:** Four in 270 images: three faint scratches called inclusion and one rolled-in scale image called pitted surface. A thin line along the rolling direction looks much like a long inclusion.

**Q:** Validation said M1 and M3 were tied at 97.04 %. Why did M1 drop on test while M2 rose?
**A:** M1 fell to 94.07 % and M2 rose from 95.93 % to 96.67 %. The validation set has 270 images, so one image is 0.37 points, and it could not separate models this close. 9 of M1's 10 pitted-surface mistakes had a confidence below 0.7, so it was unsure on these. M3's lower validation loss (0.0832 against 0.1056) had already hinted that it was the better model.

**Q:** M2 beat M1 on test. Why did you not switch finalists?
**A:** We chose M1 and M3 on validation before opening the test set. If we swapped after seeing the test, the test would become a second validation set, and its score would no longer be honest. We report M2's 96.67 % openly and keep our choice.

**Q:** Your accuracy is 98.5 %. Is that too good?
**A:** It is high because NEU is clean, balanced and taken under controlled conditions. We checked for duplicates (one pair, kept in one split) and for leakage. Our robustness test shows the limits: M3 falls to 48.1 % at the highest noise level. A real plant would need its own images.

**Q:** Why measure speed on the CPU, one image at a time?
**A:** A small computer next to the line may have no GPU, and each image must be answered as it arrives. We time 100 single-image calls after 10 warm-up calls. On Kaggle's CPU, M1 takes 21.57 ms and M3 takes 223.96 ms, about 10 times slower. M3's file is also 18 times bigger.

**Q:** M3 is more accurate, but M1 is smaller and faster. Which would you deploy?
**A:** M3 is our final model for accuracy (98.52 % against 94.07 %). M1 suits a very small computer. After TensorFlow Lite conversion M3 runs in 18.03 ms per image and fits in 2.51 MB, so it also fits a small device.

## Explainability, robustness & deployment

**Q:** How does Grad-CAM work?
**A:** We take the gradient of the predicted class score with respect to the feature maps of the last convolution layer. We average each map's gradient to get one weight per map, add up the maps weighted by these, apply ReLU, and scale the result up to the image size. Bright areas are the parts that pushed the score up.

**Q:** What did Grad-CAM show for M3 and M1?
**A:** M3's heatmaps sit on the defects: the scratch line, the inclusion streak and the dark patches. M1's often sit on the background. For inclusion, the flat steel around the defect lights up. On a scratch it missed, the background is red and the scratch is not.

**Q:** What is shortcut learning, and how did M1 show it?
**A:** The model uses an easy clue instead of the defect. Inclusion images are the flattest in our EDA, so M1 seems to have learned that a flat background means inclusion. We treat this as a strong hint, not a proof, because we only have heatmaps and the robustness results.

**Q:** Why did M1 collapse when the images got darker?
**A:** Its accuracy fell from 97.0 % to 65.9 % with images 0.15 darker, and to 31.9 % at 0.30. M3 stayed at 96.7 % at both. We think M1 relied on brightness and background texture, which change when the light changes, and not on the defect shape.

**Q:** How did you test robustness, and why on the validation set?
**A:** We added four kinds of damage at three severities each: noise, blur, darker images and lower contrast. We used the validation set because the test set is a one-time measurement. Using it again would turn it into a second validation set.

**Q:** Which model was most robust?
**A:** M3. With a 5 px blur it keeps 92.6 %, and with images 0.45 darker it keeps 91.9 %. Noise breaks every model. At standard deviation 0.10, M3 falls to 48.1 %, M1 to 24.4 % and M2 to 25.2 %.

**Q:** What does augmentation cost and what does it buy?
**A:** On clean validation images, M2 scored 100.00 % with no augmentation, 99.26 % without rotation and 95.56 % with the full set. So it costs clean accuracy. But it buys robustness to lighting. M2 keeps 89.6 % with images 0.15 darker, where M1 falls to 65.9 %. A real line has changing light, so we keep it.

**Q:** Why did rotation hurt?
**A:** It cost the most accuracy in our ablation. Scratches and rolled-in scale follow the rolling direction. Tilting the image changes that direction, so the model sees patterns that do not occur on the line. This is why we avoid 90 degree turns and keep the tilt small.

**Q:** Why did ablation and robustness use the validation set, not the test set?
**A:** We already used the test set once, in notebook 06. Choosing settings from test results would make the test score too optimistic. We did not test the best ablation network (no augmentation, 100 % on validation) on the test set, for the same reason.

**Q:** What is quantisation, and what are your size and speed numbers?
**A:** Dynamic-range quantisation stores each weight as an 8-bit integer instead of a 32-bit float, so weights take about four times less space. The file shrinks even more, because the `.keras` file also stores the optimiser's state (Adam keeps two extra numbers per weight) and the TFLite file does not. M3 went from 24.46 MB to 2.51 MB at 97.41 % validation accuracy, and 18.03 ms per image on CPU. M1 went from 1.38 MB to 0.12 MB at 97.04 %, and 10.14 ms.

**Q:** Does the quantised model give the same answers?
**A:** Almost. M3 agrees with the Keras model on 99.63 % of validation images, and M1 on 100 %. That is why we always check agreement after conversion.

**Q:** Why did the TensorFlow Lite conversion fail the first time?
**A:** M3 had random augmentation layers inside it. Their random operations (for example `StatelessRandomUniformV2` and `Tan` from the rotation) cannot be converted. We convert an inference copy without the augmentation block, which does nothing at prediction time anyway.

**Q:** What are the limits of your work?
**A:** NEU is small, clean and balanced, so 98.52 % will not carry over to a real plant. The corruptions in our robustness test are made up, not real plant images. Our speed numbers come from Kaggle's CPU only. The ablation numbers move a little between runs, so we trust the order, not the decimals.

**Q:** What would you do next?
**A:** Four things. Add noise to the augmentation, since noise at 0.10 breaks every model. Collect more and more varied data. Try a real line-scan camera and cut its long images into tiles. And test the no-augmentation BN network (100 % on validation) on new data, since we could not use our test set again.
