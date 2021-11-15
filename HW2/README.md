# Directory:

- auditory
	- classical
	- dummy
	- jobs
		- fM00223
		- sM00223
		- spm12
	- Figures


# Procedures
## Realignment
Realignment is a process that aligns each volume to a reference volume which is usually the mean image of all time points. This step is to minimize the artifacts caused by the subject's motion during data acquisition, such as head motion, etc. Head movement will lead to unwanted variation in voxels, one of the primary contributors to the lower quality of the fMRI image captured. Hence, this technique allows to correct the effect of head motion and other artifacts. 

## Coregistration
Coregistration refers to the alignment of anatomical scans and functional scans. That is, the algorithm overlays the anatomical (T1 weighted) image and the functional image and moves the images around by matching the dark areas on one image with the bright areas on the other until the mutual information is maximized. In this case, SPM takes a representative of the functional data, the mean image, as a reference and the structural image as a source. This step aligns both the anatomical and functional images into the same space and orientation and thus any warps applied to the anatomical image can be applied to the functional image.

## Segmentation
Segmentation aims to map/classify the fMRI data for individual subjects into an array of distinct tissue types according to the tissue probability map (TPM). The map defines a prior probability of finding a particular tissue type at a particular location. Here tissue types often refer to grey matter, white matter, cerebrospinal fluid, bone, soft tissue, and air/background in the SPM12 library. The purpose of segmentation is, therefore, to more accurately align different tissue types on the anatomical image during the next step of normalization.

## Normalise
Normalization is a process that aligns and warps fMRI data into a standard brain template.
This step allows to derive group statistics and improve the statistical power of the analysis. Since the analysis is conducted on a template, rather than an individual level, findings can be generalized into a population/group level and be further used to identify commonalities and differences between groups.

## Smoothing
Smoothing is a blurring process that is normally performed during image analysis tasks. This can be achieved by convolution in the spatial domain or high-pass filtering in the frequency domain. In this project, the Matlab SPM method will use a Gaussian kernel specified by Full-Width at Half Maximum (FWHM) as default. This step helps suppress noise and enhance the normality of the image data after spatial normalization. Basically, it works to improve the normalized result from the previous step and, as a result, increase the robustness of statistical testing with a lower false-positive rate and normal-like error distribution. 


# Limitations and improvements:
	(1) Data acquisition can be improved by utilizing more rigorous procedures while capturing fMRI images from patients, such as using posture correctors or other equipment to restrict patients' motion. Alternatively, shortening the time of scanning can also achieve the goal.

	(2) Since functional MRI (fMRI) sequence doesn’t acquire every slice in a volume at the same time,  we have to account for the time differences among the slices, which are typically acquired in either sequential or interleaved (acquire every other slice in each direction), where may start at the top (descending) or the bottom (ascending)

	Slice timing means correcting the time differences among the slices.

	Therefore, slice timing correction is required to compensate for this timing difference.

	Per discussion, interpolation is recommended during temporal processing. One example is linear interpolation.

	(3) Quality assurance should be conducted throughout each stage of the data preprocessing. With carefully monitoring the intermediate results, additional procedures, such as extra smoothing, should be introduced to augment the performance.


# Statistical Inference
## Figure 30.15
The result table exhibits a list of all clusters above the chosen significance level, 0.05, as well as separate (>8mm apart) maxima within a cluster. The T statistics threshold is roughly 5.25 (a more precise number, 5.254925, is provided in Figure 30.13), with a level of significance of 0.05 and degree of freedom 73. Thus, the list of clusters, to satisfy (be less than) the level of significance, must have T statistics greater than the height threshold of 5.25. As observed in the table, all clusters' p-values are less than 0.05 and their corresponding peak-level T statistics are higher than 5.25. Hence, they are identified as active voxel clusters while listening based on these specific inference parameters.

The set-level indicates that the chance of finding 13 clusters is about 0.05, as pre-specified. The cluster-level shows the 13 clusters with at least 0 voxels, as pre-specified. The peak-level presents the chance, under the null hypothesis, of finding a peak at such a location. For instance, in the cluster of 492 voxels, a peak at (57,-22,11) is being found under the null hypothesis with T-statistics of 16.74.

As indicated in the table, 3 local maxima more than 8.0mm apart have been identified. On average, the number of voxels per cluster is 0.760 and the expected number of clusters is 0.07.

## Figure 30.19
The Sections figure shows the overlay on three intersecting (sagittal, coronal, axial) slices. The highlighted regions visualize the 'listening > rest' activation over the spatially normalized, bias-corrected anatomical image. In other words, the active voxels while listening condition will be found in such areas.
