/// disease_information.dart
///
/// Complete clinical reference data for the 8 FairDermNet skin-lesion classes.
///
/// DISCLAIMER: This information is for AI-assisted screening and decision
/// support only. A prediction from this app is NOT a medical diagnosis.
/// Suspicious lesions must be assessed by a qualified dermatologist.

library disease_information;

// ─────────────────────────────────────────────────────────────────────────────
// Enums
// ─────────────────────────────────────────────────────────────────────────────

enum DiseaseStatus { malignant, precancerous, benign, variable }

enum RiskLevel { high, medium, low }

// ─────────────────────────────────────────────────────────────────────────────
// DiseaseInfo model
// ─────────────────────────────────────────────────────────────────────────────

class DiseaseInfo {
  /// Full display name, e.g. "Melanoma"
  final String name;

  /// Short class ID used by the model, e.g. "MEL"
  final String classId;

  final DiseaseStatus status;
  final RiskLevel riskLevel;

  /// One-paragraph overview shown in the main prediction card.
  final String overview;

  /// Bullet-point symptoms / clinical signs.
  final List<String> symptoms;

  /// Bullet-point visual / dermoscopic features.
  final List<String> visualFeatures;

  /// Diagnostic steps.
  final List<String> diagnosis;

  /// What the patient should do.
  final List<String> whatToDo;

  /// Reference links: list of maps with 'label' and 'url'.
  final List<Map<String, String>> references;

  const DiseaseInfo({
    required this.name,
    required this.classId,
    required this.status,
    required this.riskLevel,
    required this.overview,
    required this.symptoms,
    required this.visualFeatures,
    required this.diagnosis,
    required this.whatToDo,
    required this.references,
  });

  // ── Convenience getters ────────────────────────────────────────────────────

  String get statusLabel {
    switch (status) {
      case DiseaseStatus.malignant:
        return 'Malignant';
      case DiseaseStatus.precancerous:
        return 'Precancerous';
      case DiseaseStatus.benign:
        return 'Benign';
      case DiseaseStatus.variable:
        return 'Usually Benign / Variable';
    }
  }

  String get riskLabel {
    switch (riskLevel) {
      case RiskLevel.high:
        return 'High';
      case RiskLevel.medium:
        return 'Medium';
      case RiskLevel.low:
        return 'Low';
    }
  }
}

// ─────────────────────────────────────────────────────────────────────────────
// Dataset — all 8 classes
// ─────────────────────────────────────────────────────────────────────────────

const List<DiseaseInfo> kDiseaseData = [
  // ── 0 · Melanoma ──────────────────────────────────────────────────────────
  DiseaseInfo(
    name: 'Melanoma',
    classId: 'MEL',
    status: DiseaseStatus.malignant,
    riskLevel: RiskLevel.high,
    overview:
        'Melanoma is a malignant skin cancer that develops from melanocytes, '
        'the cells responsible for producing skin pigment. It may develop in an '
        'existing mole or appear as a new skin lesion. Early detection is '
        'important because melanoma can invade surrounding tissue and spread to '
        'other parts of the body.',
    symptoms: [
      'New or changing mole or pigmented lesion',
      'Asymmetrical shape',
      'Irregular or poorly defined borders',
      'Multiple or uneven colors',
      'Increase in size',
      'Changes in shape, color, or appearance over time',
      'Itching, bleeding, crusting, or tenderness',
    ],
    visualFeatures: [
      'Asymmetry',
      'Irregular borders',
      'Multiple colors',
      'Uneven pigmentation',
      'Irregular / chaotic structure',
      'Noticeable difference from surrounding moles',
    ],
    diagnosis: [
      'Clinical examination',
      'Dermoscopy',
      'Skin biopsy',
      'Histopathological examination of the biopsy',
      'Additional staging investigations when melanoma is confirmed',
    ],
    whatToDo: [
      'Seek evaluation by a dermatologist promptly.',
      'Do not attempt to remove or treat a suspicious lesion yourself.',
      'Monitor lesions for changes while awaiting professional assessment.',
    ],
    references: [
      {
        'label': 'American Academy of Dermatology — Melanoma',
        'url': 'https://www.aad.org/public/diseases/skin-cancer/types/melanoma',
      },
      {
        'label': 'National Cancer Institute — Melanoma',
        'url':
            'https://www.cancer.gov/types/skin/patient/melanoma-treatment-pdq',
      },
    ],
  ),

  // ── 1 · Basal Cell Carcinoma ──────────────────────────────────────────────
  DiseaseInfo(
    name: 'Basal Cell Carcinoma',
    classId: 'BCC',
    status: DiseaseStatus.malignant,
    riskLevel: RiskLevel.high,
    overview:
        'Basal cell carcinoma is the most common type of skin cancer. It usually '
        'grows slowly and rarely spreads to distant organs, but an untreated '
        'lesion can grow into nearby tissues and cause significant local damage.',
    symptoms: [
      'Shiny or pearly bump',
      'Firm raised growth',
      'Pink, red, brown, or skin-colored lesion',
      'Persistent scaly patch',
      'Open sore that does not heal',
      'Sore that repeatedly heals and returns',
      'Bleeding or crusting',
      'Gradually enlarging lesion',
    ],
    visualFeatures: [
      'Pearly or translucent appearance',
      'Shiny surface',
      'Raised borders',
      'Ulceration',
      'Visible small blood vessels',
      'Pigmentation in some lesions',
    ],
    diagnosis: [
      'Clinical skin examination',
      'Dermoscopy',
      'Skin biopsy',
      'Histopathological examination',
    ],
    whatToDo: [
      'Arrange a dermatological evaluation, particularly for a lesion that persists or continues to grow.',
      'Seek assessment sooner if it repeatedly bleeds, crusts, or fails to heal.',
    ],
    references: [
      {
        'label': 'American Academy of Dermatology — Basal Cell Carcinoma',
        'url':
            'https://www.aad.org/public/diseases/skin-cancer/types/common/bcc',
      },
      {
        'label': 'National Cancer Institute — Skin Cancer',
        'url': 'https://www.cancer.gov/types/skin',
      },
    ],
  ),

  // ── 2 · Squamous Cell Carcinoma ───────────────────────────────────────────
  DiseaseInfo(
    name: 'Squamous Cell Carcinoma',
    classId: 'SCC',
    status: DiseaseStatus.malignant,
    riskLevel: RiskLevel.high,
    overview:
        'Squamous cell carcinoma is a skin cancer arising from keratinocytes, '
        'the predominant cells in the outer layer of the skin. It commonly '
        'occurs on areas exposed to ultraviolet radiation and can sometimes '
        'develop from actinic keratosis.',
    symptoms: [
      'Rough, scaly patch',
      'Firm bump or nodule',
      'Red or inflamed lesion',
      'Wart-like growth',
      'Persistent sore',
      'Non-healing wound',
      'Crusting or bleeding',
      'Rapid growth',
      'Pain or tenderness',
      'Itching, burning, or stinging',
    ],
    visualFeatures: [
      'Scaly or keratotic surface',
      'Red or inflamed appearance',
      'Firm raised lesion',
      'Crusting',
      'Ulceration',
      'Irregular growth',
    ],
    diagnosis: [
      'Clinical examination',
      'Dermoscopy when appropriate',
      'Skin biopsy',
      'Histopathological examination',
    ],
    whatToDo: [
      'Have persistent, rapidly growing, painful, bleeding, or non-healing lesions evaluated by a dermatologist.',
      'Do not assume that a lesion is harmless based only on its appearance.',
    ],
    references: [
      {
        'label': 'American Academy of Dermatology — Squamous Cell Carcinoma',
        'url':
            'https://www.aad.org/public/diseases/skin-cancer/types/common/scc',
      },
      {
        'label': 'National Cancer Institute — Skin Cancer',
        'url': 'https://www.cancer.gov/types/skin',
      },
    ],
  ),

  // ── 3 · Actinic Keratosis ─────────────────────────────────────────────────
  DiseaseInfo(
    name: 'Actinic Keratosis',
    classId: 'AK',
    status: DiseaseStatus.precancerous,
    riskLevel: RiskLevel.medium,
    overview:
        'Actinic keratosis is a precancerous skin lesion caused primarily by '
        'cumulative ultraviolet (UV) exposure. It can sometimes progress to '
        'squamous cell carcinoma, so persistent or suspicious lesions should '
        'be evaluated by a dermatologist.',
    symptoms: [
      'Rough, sandpaper-like patch',
      'Scaly or crusty area',
      'Red, pink, brown, gray, or skin-colored lesion',
      'Flat or slightly raised lesion',
      'Dry or scaly area on the lips',
      'Itching',
      'Burning or stinging',
      'Tenderness or pain',
      'Occasional bleeding',
    ],
    visualFeatures: [
      'Rough / scaly surface',
      'Flat or slightly elevated lesion',
      'Red or pink coloration',
      'Keratotic appearance',
      'Irregular patches',
      'May occur as multiple lesions in sun-damaged skin',
    ],
    diagnosis: [
      'Clinical examination',
      'Dermoscopy when required',
      'Biopsy when the lesion is atypical or squamous cell carcinoma is suspected',
      'Histopathological examination when biopsied',
    ],
    whatToDo: [
      'Arrange dermatological evaluation.',
      'Protect the affected skin from further UV exposure.',
      'Pay particular attention to lesions that become thick, painful, rapidly changing, or bleed.',
    ],
    references: [
      {
        'label': 'American Academy of Dermatology — Actinic Keratosis Symptoms',
        'url':
            'https://www.aad.org/public/diseases/a-z/actinic-keratosis-symptoms',
      },
      {
        'label':
            'American Academy of Dermatology — Actinic Keratosis Treatment',
        'url':
            'https://www.aad.org/public/diseases/a-z/actinic-keratosis-treatment',
      },
    ],
  ),

  // ── 4 · Melanocytic Nevi ──────────────────────────────────────────────────
  DiseaseInfo(
    name: 'Melanocytic Nevi',
    classId: 'NV',
    status: DiseaseStatus.benign,
    riskLevel: RiskLevel.low,
    overview:
        'Melanocytic nevi, commonly called moles, are benign growths formed by '
        'melanocytes. They may be present from birth or develop later in life '
        'and can be flat or raised.',
    symptoms: [
      'Usually small and well-defined',
      'Round or oval shape',
      'Symmetrical appearance',
      'Relatively uniform color',
      'May be flat or raised',
      'Usually remain stable over time',
    ],
    visualFeatures: [
      'Symmetry',
      'Regular borders',
      'Relatively uniform pigmentation',
      'Well-circumscribed structure',
      'May be brown, black, pink, or skin-colored',
    ],
    diagnosis: [
      'Clinical examination',
      'Dermoscopy when necessary',
      'Comparison with other moles',
      'Digital monitoring in selected cases',
      'Biopsy or excision if melanoma is suspected',
    ],
    whatToDo: [
      'Usually no treatment is necessary for an uncomplicated benign nevus.',
      'Monitor for changes in size, shape, color, or structure.',
      'Seek dermatological evaluation if a mole changes or looks significantly different from other moles.',
    ],
    references: [
      {
        'label': 'DermNet — Melanocytic Naevus',
        'url': 'https://dermnetnz.org/topics/melanocytic-naevus',
      },
      {
        'label': 'American Academy of Dermatology — Moles',
        'url': 'https://www.aad.org/public/diseases/a-z/moles-overview',
      },
    ],
  ),

  // ── 5 · Benign Keratosis ──────────────────────────────────────────────────
  DiseaseInfo(
    name: 'Benign Keratosis',
    classId: 'BKL',
    status: DiseaseStatus.benign,
    riskLevel: RiskLevel.low,
    overview:
        'Benign keratosis is a category of non-cancerous keratotic or '
        'keratosis-like skin lesions. In the HAM10000 taxonomy used by this '
        'model, this category includes seborrheic keratoses, solar lentigines, '
        'and lichen-planus-like keratoses. BKL is therefore an umbrella class '
        'rather than a single disease.',
    symptoms: [
      'Brown, tan, gray, or black lesion',
      'Flat or raised appearance',
      'Rough or waxy surface',
      'Scaly or warty texture',
      '"Stuck-on" appearance in typical seborrheic keratoses',
      'Usually painless',
      'May become itchy or irritated',
    ],
    visualFeatures: [
      'Well-defined borders',
      'Waxy or keratotic surface',
      'Pigmentation',
      'Scaly texture',
      '"Stuck-on" appearance for seborrheic keratosis',
      'Variable morphology depending on subtype',
    ],
    diagnosis: [
      'Clinical examination',
      'Dermoscopy',
      'Biopsy or removal when the appearance is atypical or malignancy cannot be excluded',
    ],
    whatToDo: [
      'Usually benign and does not require treatment.',
      'Have the lesion examined if it changes rapidly, bleeds, becomes symptomatic, or looks different from typical lesions.',
    ],
    references: [
      {
        'label': 'American Academy of Dermatology — Seborrheic Keratoses',
        'url':
            'https://www.aad.org/public/diseases/a-z/seborrheic-keratoses-overview',
      },
      {
        'label': 'DermNet — Seborrhoeic Keratosis',
        'url': 'https://dermnetnz.org/topics/seborrhoeic-keratosis',
      },
      {
        'label': 'HAM10000 Dataset Paper',
        'url': 'https://doi.org/10.1038/sdata.2018.161',
      },
    ],
  ),

  // ── 6 · Dermatofibroma ────────────────────────────────────────────────────
  DiseaseInfo(
    name: 'Dermatofibroma',
    classId: 'DF',
    status: DiseaseStatus.benign,
    riskLevel: RiskLevel.low,
    overview:
        'Dermatofibroma is a common benign skin nodule composed of fibrous '
        'tissue. It frequently occurs on the lower legs and generally remains '
        'stable over time.',
    symptoms: [
      'Small, firm skin nodule',
      'Brown, tan, reddish, or darker coloration',
      'Commonly occurs on the legs',
      'Usually stable',
      'May be slightly raised',
      'Often produces a central depression when pinched',
    ],
    visualFeatures: [
      'Firm, well-defined papule or nodule',
      'Central pale area',
      'Peripheral pigmentation',
      'Brown or reddish coloration',
      'Characteristic "dimple" sign in some lesions',
    ],
    diagnosis: [
      'Clinical examination',
      'Palpation',
      'Dermoscopy',
      'Biopsy if the lesion is atypical or the diagnosis is uncertain',
    ],
    whatToDo: [
      'Usually requires no treatment.',
      'Seek medical evaluation if it changes rapidly, becomes painful, bleeds, or develops unusual characteristics.',
    ],
    references: [
      {
        'label': 'DermNet — Dermatofibroma',
        'url': 'https://dermnetnz.org/topics/dermatofibroma',
      },
      {
        'label': 'DermNet — Dermatofibroma Dermoscopy',
        'url': 'https://dermnetnz.org/topics/dermatofibroma-dermoscopy',
      },
    ],
  ),

  // ── 7 · Vascular Lesion ───────────────────────────────────────────────────
  DiseaseInfo(
    name: 'Vascular Lesion',
    classId: 'VASC',
    status: DiseaseStatus.variable,
    riskLevel: RiskLevel.low,
    overview:
        'Vascular lesions are lesions associated with blood vessels and may '
        'appear red, purple, blue, or dark because of blood within or beneath '
        'the skin. This is an umbrella class. In the HAM10000 taxonomy, VASC '
        'includes angiomas, angiokeratomas, pyogenic granulomas, and '
        'hemorrhagic lesions.',
    symptoms: [
      'Red, purple, blue, or dark coloration',
      'Flat or raised lesion',
      'Small papule or patch',
      'May be soft or firm depending on the lesion',
      'Some lesions may bleed after minor trauma',
      'Pyogenic granulomas may grow rapidly and bleed easily',
    ],
    visualFeatures: [
      'Red / purple coloration',
      'Vascular structures',
      'Well-defined papules or patches',
      'Red-blue areas or clods',
      'Possible ulceration or bleeding in some lesions',
    ],
    diagnosis: [
      'Clinical examination',
      'Dermoscopy',
      'Assessment of growth and symptoms',
      'Biopsy when the lesion is unusual or the diagnosis is uncertain',
    ],
    whatToDo: [
      'Many vascular lesions are benign and require no treatment.',
      'Seek dermatological evaluation for rapidly growing lesions, repeated bleeding, significant changes, or unusual appearances.',
    ],
    references: [
      {
        'label': 'DermNet — Cherry Angioma',
        'url': 'https://dermnetnz.org/topics/cherry-angioma',
      },
      {
        'label': 'DermNet — Pyogenic Granuloma',
        'url': 'https://dermnetnz.org/topics/pyogenic-granuloma',
      },
      {
        'label': 'HAM10000 Dataset Paper',
        'url': 'https://doi.org/10.1038/sdata.2018.161',
      },
    ],
  ),
];

// ─────────────────────────────────────────────────────────────────────────────
// Lookup helpers
// ─────────────────────────────────────────────────────────────────────────────

/// Returns the [DiseaseInfo] whose [name] matches [label] (case-insensitive).
/// Falls back to the first entry if no match is found.
DiseaseInfo diseaseInfoForLabel(String label) {
  return kDiseaseData.firstWhere(
    (d) => d.name.toLowerCase() == label.toLowerCase(),
    orElse: () => kDiseaseData.first,
  );
}

/// Returns the [DiseaseInfo] whose [classId] matches [id] (case-insensitive).
DiseaseInfo? diseaseInfoForClassId(String id) {
  try {
    return kDiseaseData.firstWhere(
      (d) => d.classId.toLowerCase() == id.toLowerCase(),
    );
  } catch (_) {
    return null;
  }
}
