
## Index	Class	Short	Risk tier
0	Melanoma	MEL	High (malignant)
1	Basal Cell Carcinoma	BCC	High (malignant)
2	Squamous Cell Carcinoma	SCC	High (malignant)
3	Actinic Keratosis	AK	Medium (pre-cancerous)
4	Melanocytic Nevi	NV	Low (benign)
5	Benign Keratosis	BKL	Low (benign)
6	Dermatofibroma	DF	Low (benign)
7	Vascular Lesion	VASC	Low (benign)
8	Merkel Cell Carcinoma	MCC	High (malignant)
9	Sebaceous Carcinoma	SC	High (malignant)
10	Dermatofibrosarcoma Protuberans	DFSP	High (malignant)
11	Cutaneous T-Cell Lymphoma	CTCL	High (malignant)
12	Keratoacanthoma	KA	Medium (low-grade malignant / rapid-growth)
13	Bowen's Disease (Squamous Carcinoma in situ)	BD	Medium (pre-cancerous / early malignancy)
14	Seborrheic Keratosis	SK	Low (benign)
15	Cherry Angioma	CA	Low (benign)


1. The HAM10000 Dataset (7 Classes)

This is the most famous skin lesion benchmark dataset on Kaggle. It contains 10,015 dermoscopic images and covers exactly 7 of the classes on your list: [1] (https://www.kaggle.com/datasets/kmader/skin-cancer-mnist-ham10000), [2] (https://skinive.com/skin-disease-datasets/)
• 0: MEL (Melanoma)
• 1: BCC (Basal Cell Carcinoma)
• 3: AK (Actinic Keratosis - labeled as akiec)
• 4: NV (Melanocytic Nevi)
• 5: BKL (Benign Keratosis)
• 6: DF (Dermatofibroma)
• 7: VASC (Vascular Lesion) [1] (https://data.mendeley.com/datasets/hpcf9psdy7), [2] (https://www.kaggle.com/datasets/kmader/skin-cancer-mnist-ham10000)
Where to get it: You can download it instantly via the Kaggle HAM10000 Page. [1] (https://www.kaggle.com/datasets/kmader/skin-cancer-mnist-ham10000)

2. The ISIC Archive Challenges (Adds 3 Classes)

The International Skin Imaging Collaboration (ISIC) hosts the world's largest repository of skin images. Their larger dataset competitions (like ISIC 2019 and ISIC 2024) include everything in HAM10000 plus: [1] (https://challenge.isic-archive.com/), [2] (https://workshop.isic-archive.com/2025/), [3] (https://api.isic-archive.com/collections/)
• 2: SCC (Squamous Cell Carcinoma)
• 13: BD (Bowen's Disease / Squamous Carcinoma in situ)
• 14: SK (Seborrheic Keratosis - often grouped under BKL or split out) [1] (https://api.isic-archive.com/collections/), [2] (https://www.kaggle.com/datasets/kmader/skin-cancer-mnist-ham10000)
Where to get it: Downloadable via Kaggle competition data or the ISIC Archive API. [1] (https://skinive.com/skin-disease-datasets/), [2] (https://api.isic-archive.com/collections/)

3. General Dermatology & Rare Cancer Datasets (The Rest)

For the rare cancers (8: MCC, 9: SC, 10: DFSP, 11: CTCL) and standard clinical spots (12: KA, 15: CA), you have to look into broader multi-class clinical datasets because they are too rare to fit into standard mole-focused sets: [1] (https://workshop.isic-archive.com/2025/), [2] (https://skinive.com/skin-disease-datasets/)
• Fitzpatrick 17k Dataset: Great for clinical (non-dermoscopic) images of rare conditions across different skin tones.
• DermNet Dataset: A general clinical dataset available on Kaggle containing tens of thousands of images across 23+ categories, where you will find conditions like Cutaneous Lymphoma (CTCL) and Cherry Angiomas. [1] (https://www.kaggle.com/shubhamgoel27/dermnet/kernels), [2] (https://workshop.isic-archive.com/2025/), [3] (https://www.kaggle.com/datasets/pacificrm/skindiseasedataset)

## Where to get it: 
- Available by searching the Kaggle Dermnet Dataset or using the Harvard Dataverse for Fitzpatrick 17k. [1] (https://www.kaggle.com/shubhamgoel27/dermnet/kernels)[2] (https://data.mendeley.com/datasets/hpcf9psdy7)
