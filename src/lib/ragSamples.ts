// Built-in shared practice datasets for the RAG Lab.
//
// STORAGE NOTE: these live in the app bundle (the repo), NOT in cloud object
// storage. One copy serves every student, and no per-student uploads or
// embeddings are persisted — experiments store only configuration + metrics.
// That's the whole point: 100 students can practice without spending the 1 GB
// shared storage budget.
//
// Each dataset ships with a few suggested questions (with a plausible "expected"
// phrase) so students have something concrete to retrieve for.

export type RagSample = {
  id: string;
  name: string;
  emoji: string;
  description: string;
  questions: string[];
  text: string;
};

const HANDBOOK = `Northbridge College Student Handbook — 2026 Edition

Attendance. Students must attend at least 75% of scheduled sessions in each course to be eligible for the final examination. Attendance below 75% results in a "detained" status, and the student must repeat the course. Medical absences are excused only when a certificate is submitted to the department office within seven days of returning to campus.

Grading. Final grades are computed as 40% continuous assessment and 60% final examination. The passing grade for undergraduate courses is 40 out of 100. A grade of "I" (Incomplete) may be granted at the instructor's discretion and must be cleared within one semester, after which it converts to an "F".

Refunds. A student who withdraws within the first two weeks of a semester receives a 90% refund of tuition. Withdrawal in weeks three and four returns 50%. After the fourth week, no tuition refund is issued. The one-time registration fee of $150 is non-refundable in all cases.

Library. The main library is open from 8am to 10pm on weekdays and 9am to 5pm on Saturdays. It is closed on Sundays and public holidays. Books may be borrowed for 14 days and renewed twice. The fine for an overdue book is $0.50 per day, capped at the replacement cost of the book.

Housing. On-campus residence is guaranteed for first-year students who apply before July 1. Residence fees are billed per semester and include utilities but not meals. A refundable security deposit of $300 is collected at move-in and returned within 30 days of move-out, less any damage charges.

Academic integrity. Plagiarism, unauthorized collaboration, and the use of prohibited materials during examinations are violations of the academic integrity code. A first offense typically results in a zero for the assessment; a second offense may lead to suspension for one semester.

Graduation. To graduate, an undergraduate must complete 120 credit hours with a cumulative GPA of at least 2.0. Applications to graduate are due eight weeks before the end of the final semester.`;

const PYTHON = `Python Essentials — Quick Reference Notes

Variables and types. Python is dynamically typed: a variable takes the type of the value assigned to it. The core built-in types are int, float, str, bool, list, tuple, dict, and set. Use type(x) to inspect a value's type at runtime. Strings are immutable; lists are mutable.

Lists vs tuples. A list is mutable and written with square brackets: [1, 2, 3]. A tuple is immutable and written with parentheses: (1, 2, 3). Because tuples are immutable they can be used as dictionary keys, while lists cannot. Both support indexing and slicing.

Dictionaries. A dict maps keys to values and is written {"a": 1, "b": 2}. Access a value with d["a"], or d.get("a", default) to avoid a KeyError when the key is missing. Iterating a dict yields its keys; use d.items() to iterate key-value pairs.

Functions. Define a function with def. Arguments may have default values, and you can pass them positionally or by keyword. Use *args to accept extra positional arguments and **kwargs for extra keyword arguments. A function returns None if it has no explicit return statement.

Comprehensions. A list comprehension builds a list in one expression: [x*x for x in range(5)] produces [0, 1, 4, 9, 16]. There are also dict and set comprehensions. Comprehensions are usually faster and more readable than an equivalent for-loop that appends.

Exceptions. Wrap risky code in try/except. Catch specific exceptions (except ValueError) rather than a bare except, which also swallows KeyboardInterrupt. The finally block always runs, whether or not an exception occurred, and is used for cleanup such as closing a file.

Virtual environments. Create an isolated environment with python -m venv .venv and activate it before installing packages with pip. This keeps each project's dependencies separate. A requirements.txt file, generated with pip freeze, records the exact versions so the environment can be reproduced.`;

const ML = `Machine Learning Study Notes

Supervised vs unsupervised. In supervised learning the training data has labels, and the model learns to predict the label for new inputs — classification predicts a category, regression predicts a number. In unsupervised learning there are no labels; the model finds structure, as in clustering or dimensionality reduction.

Train/test split. Data is split so the model is trained on one portion and evaluated on unseen data. A common split is 80% training and 20% testing. Cross-validation goes further, rotating which fold is held out, giving a more reliable estimate of performance on small datasets.

Overfitting and underfitting. An overfit model memorizes the training data and performs poorly on new data — it has low training error but high test error. An underfit model is too simple to capture the pattern and performs poorly on both. Regularization, more data, and simpler models reduce overfitting.

Bias-variance tradeoff. Bias is error from wrong assumptions; variance is error from sensitivity to the training set. Increasing model complexity lowers bias but raises variance. The goal is the sweet spot that minimizes total error on unseen data.

Evaluation metrics. For classification, accuracy is the fraction correct, but it misleads on imbalanced data. Precision is the fraction of positive predictions that are correct; recall is the fraction of actual positives that were found. The F1 score is their harmonic mean. For regression, common metrics are mean squared error and R-squared.

Gradient descent. Many models are trained by minimizing a loss function with gradient descent, which repeatedly nudges parameters in the direction that reduces loss. The learning rate controls step size: too high and training diverges, too low and it converges slowly.

Feature scaling. Algorithms that rely on distances or gradients — such as k-nearest neighbors, SVMs, and neural networks — benefit from scaling features to a similar range, using standardization (zero mean, unit variance) or min-max normalization. Tree-based models generally do not require scaling.`;

const MANUAL = `AeroBrew X1 Coffee Maker — User Manual

Setup. Before first use, rinse the water tank and run one full brew cycle with water only, no coffee grounds. The tank holds 1.2 liters, enough for about ten cups. Always use cold, filtered water for the best flavor.

Brewing. Add one level scoop of ground coffee per cup. Close the lid until it clicks, select the cup count, and press Start. The default brew temperature is 92 degrees Celsius. A single cup takes about one minute; a full carafe takes about eight minutes.

Strength control. The Strength button cycles through Mild, Medium, and Bold. Bold mode slows the water flow so it spends longer in contact with the grounds, producing a stronger cup. The selected strength is remembered until the machine is unplugged.

Cleaning. Rinse the removable filter basket after every use. Descale the machine every two months, or more often in hard-water areas, using a mix of equal parts white vinegar and water run through a full cycle, followed by two cycles of plain water. Do not put the water tank in a dishwasher.

Auto-off. To save energy the machine switches off automatically after 30 minutes of inactivity. The keep-warm plate holds the carafe at serving temperature during that time. You can disable auto-off by holding the Start button for five seconds.

Troubleshooting. If the machine will not turn on, check that the tank is seated correctly — a safety switch prevents operation when it is removed. If brewing is slow, the machine likely needs descaling. Three blinking lights indicate the tank is empty.

Warranty. The AeroBrew X1 is covered by a two-year limited warranty against manufacturing defects. The warranty does not cover damage from limescale build-up caused by skipped descaling, which is considered improper maintenance.`;

export const RAG_SAMPLES: RagSample[] = [
  {
    id: "college-handbook", name: "College Handbook", emoji: "🎓",
    description: "Student policies — attendance, grading, refunds, library, housing.",
    questions: ["What attendance is required to sit the final exam?", "How much tuition is refunded if I withdraw in week three?", "What GPA do I need to graduate?"],
    text: HANDBOOK,
  },
  {
    id: "python-notes", name: "Python Notes", emoji: "🐍",
    description: "Python basics — types, lists vs tuples, functions, exceptions, venvs.",
    questions: ["What is the difference between a list and a tuple?", "How do I avoid a KeyError when reading a dict?", "How do I create a virtual environment?"],
    text: PYTHON,
  },
  {
    id: "ml-notes", name: "Machine Learning Notes", emoji: "🤖",
    description: "ML concepts — supervised vs unsupervised, overfitting, metrics, gradient descent.",
    questions: ["What is the difference between precision and recall?", "What causes overfitting and how do you reduce it?", "Which models need feature scaling?"],
    text: ML,
  },
  {
    id: "product-manual", name: "Product Manual", emoji: "📘",
    description: "AeroBrew X1 coffee maker — setup, brewing, cleaning, warranty.",
    questions: ["How often should I descale the machine?", "What does the warranty not cover?", "How do I disable auto-off?"],
    text: MANUAL,
  },
];

export function ragSampleList() {
  return RAG_SAMPLES.map((s) => ({
    id: s.id, name: s.name, emoji: s.emoji, description: s.description,
    questions: s.questions, words: s.text.split(/\s+/).filter(Boolean).length,
  }));
}

export function ragSampleById(id: string): RagSample | undefined {
  return RAG_SAMPLES.find((s) => s.id === id);
}
