import matplotlib.pyplot as plt

from assessor_functions import *

lrw_n_of_frames_per_sample_train = np.array(load_array_of_var_per_sample_from_csv(csv_file_name=N_OF_FRAMES_PER_SAMPLE_CSV_FILE, collect_type='train', collect_by='sample'))
lrw_n_of_frames_per_sample_val = np.array(load_array_of_var_per_sample_from_csv(csv_file_name=N_OF_FRAMES_PER_SAMPLE_CSV_FILE, collect_type='val', collect_by='sample'))
lrw_n_of_frames_per_sample_test = np.array(load_array_of_var_per_sample_from_csv(csv_file_name=N_OF_FRAMES_PER_SAMPLE_CSV_FILE, collect_type='test', collect_by='sample'))

lrw_lipreader_dense_train, lrw_lipreader_softmax_train, lrw_correct_one_hot_y_arg_train = load_dense_softmax_y(collect_type='train')
lrw_lipreader_dense_val, lrw_lipreader_softmax_val, lrw_correct_one_hot_y_arg_val = load_dense_softmax_y(collect_type='val')
lrw_lipreader_dense_test, lrw_lipreader_softmax_test, lrw_correct_one_hot_y_arg_test = load_dense_softmax_y(collect_type='test')

lrw_lipreader_correct_or_wrong_train = np.argmax(lrw_lipreader_softmax_train, axis=1) == lrw_correct_one_hot_y_arg_train
lrw_lipreader_correct_or_wrong_val = np.argmax(lrw_lipreader_softmax_val, axis=1) == lrw_correct_one_hot_y_arg_val
lrw_lipreader_correct_or_wrong_test = np.argmax(lrw_lipreader_softmax_test, axis=1) == lrw_correct_one_hot_y_arg_test

# LIPREADER PREDS vs N OF FRAMES

range_of_n_of_frames = np.arange(lrw_n_of_frames_per_sample_val.min(), lrw_n_of_frames_per_sample_val.max()+1)

def eval(range_of_feature, lrw_feature, lrw_lipreader_correct_or_wrong):
    num_of_samples = []
    num_of_correct_in_samples = []
    for n in range_of_feature:
        bool_of_samples_of_n_frames = lrw_feature == n
        num_of_samples_of_n_frames = np.sum(bool_of_samples_of_n_frames)
        num_of_correct_in_samples_of_n_frames = np.sum(lrw_lipreader_correct_or_wrong[bool_of_samples_of_n_frames])
        num_of_samples.append(num_of_samples_of_n_frames)
        num_of_correct_in_samples.append(num_of_correct_in_samples_of_n_frames)
    return num_of_samples, num_of_correct_in_samples

num_of_samples_val, num_of_correct_in_samples_val = eval(range_of_n_of_frames, lrw_n_of_frames_per_sample_val, lrw_lipreader_correct_or_wrong_val)
num_of_samples_test, num_of_correct_in_samples_test = eval(range_of_n_of_frames, lrw_n_of_frames_per_sample_test, lrw_lipreader_correct_or_wrong_test)


def plot_samples_correct_predictions(range_of_feature, num_of_samples, num_of_correct_in_samples, collect_type="val", x_label="# of frames in sample)"):
    percentage_of_correct_samples = np.array(num_of_correct_in_samples)/(np.array(num_of_samples) + 1e-8)*100
    plt.subplot(121)
    plt.bar(range_of_feature, num_of_samples, label="Total samples")
    plt.bar(range_of_feature, num_of_correct_in_samples, label="Lipreader correct predictions")
    plt.title("LRW " + collect_type + " total samples,\n# of correct predictions")
    plt.legend(fontsize=7)
    plt.xlabel(x_label)
    plt.subplot(122)
    plt.scatter(range_of_feature, percentage_of_correct_samples)
    plt.yticks(np.arange(0, 101, 10))
    plt.gca().yaxis.grid(True)
    plt.xlabel(x_label)
    plt.title("% of samples predicted\ncorrectly by lipreader")
    plt.show()

plot_samples_correct_predictions(range_of_n_of_frames, num_of_samples_val, num_of_correct_in_samples_val, collect_type="val", x_label="# of frames in sample")
plot_samples_correct_predictions(range_of_n_of_frames, num_of_samples_test, num_of_correct_in_samples_test, collect_type="test", x_label="# of frames in sample")


# LIPREADER PREDS vs BILABIALS

b_or_not = np.repeat(np.load('bilabial_or_not.npy'), 50)

percentage_lipreader_correct_among_bilabials = np.mean(lrw_lipreader_correct_or_wrong_val[b_or_not == 1])*100
percentage_lipreader_correct_among_non_bilabials = np.mean(lrw_lipreader_correct_or_wrong_val[b_or_not == 0])*100
# >>> percentage_lipreader_correct_among_bilabials
# 74.76683937823833
# >>> percentage_lipreader_correct_among_non_bilabials
# 67.407166123778509

range_of_bilabials = np.arange(b_or_not.min(), b_or_not.max()+1)

n_of_b_or_not_val, n_of_correct_in_b_or_not_val = eval(range_of_bilabials, b_or_not, lrw_lipreader_correct_or_wrong_val)
n_of_b_or_not_test, n_of_correct_in_b_or_not_test = eval(range_of_bilabials, b_or_not, lrw_lipreader_correct_or_wrong_test)

plot_samples_correct_predictions(range_of_bilabials, n_of_b_or_not_val, n_of_correct_in_b_or_not_val, collect_type="val", x_label="bilabial (1) or not (0)")
plot_samples_correct_predictions(range_of_bilabials, n_of_b_or_not_test, n_of_correct_in_b_or_not_test, collect_type="test", x_label="bilabial (1) or not (0)")


# LIPREADER PREDS vs SYLLABLES

n_of_syllables = np.repeat(np.load('n_of_syllables.npy'), 50)

range_of_syllables = np.arange(n_of_syllables.min(), n_of_syllables.max()+1)

n_of_syllables_val, n_of_correct_in_syllables_val = eval(range_of_syllables, n_of_syllables, lrw_lipreader_correct_or_wrong_val)
n_of_syllables_test, n_of_correct_in_syllables_test = eval(range_of_syllables, n_of_syllables, lrw_lipreader_correct_or_wrong_test)

plot_samples_correct_predictions(range_of_syllables, n_of_syllables_val, n_of_correct_in_syllables_val, collect_type="val", x_label="n_of_syllables")
plot_samples_correct_predictions(range_of_syllables, n_of_syllables_test, n_of_correct_in_syllables_test, collect_type="test", x_label="n_of_syllables")


# LIPREADER PREDS vs SOFTMAX RATIOS

def calc_softmax_ratios(lrw_lipreader_softmax):
    lrw_lipreader_softmax_sorted = np.sort(lrw_lipreader_softmax, axis=1)[:, ::-1]
    lrw_lipreader_softmax_sorted_ratios = lrw_lipreader_softmax_sorted[:, :5] / lrw_lipreader_softmax_sorted[:, 1:6]
    lrw_lipreader_softmax_sorted_ratios = lrw_lipreader_softmax_sorted_ratios / np.reshape(np.sum(lrw_lipreader_softmax_sorted_ratios, axis=1), (len(lrw_lipreader_softmax_sorted_ratios), 1))
    return lrw_lipreader_softmax_sorted_ratios

lrw_lipreader_softmax_train_sorted_ratios = calc_softmax_ratios(lrw_lipreader_softmax_train)
lrw_lipreader_softmax_val_sorted_ratios = calc_softmax_ratios(lrw_lipreader_softmax_val)
lrw_lipreader_softmax_test_sorted_ratios = calc_softmax_ratios(lrw_lipreader_softmax_test)

lrw_lipreader_softmax_train_sorted_ratios = np.array(lrw_lipreader_softmax_train_sorted_ratios*100, dtype=int) / 100.
lrw_lipreader_softmax_val_sorted_ratios = np.array(lrw_lipreader_softmax_val_sorted_ratios*100, dtype=int) / 100.
lrw_lipreader_softmax_test_sorted_ratios = np.array(lrw_lipreader_softmax_test_sorted_ratios*100, dtype=int) / 100.

range_of_softmax_ratios = np.arange(0, 1.01, .01)

# 1st ratio
n_of_softmax_ratios_1_train, n_of_correct_in_softmax_ratios_1_train = eval(range_of_softmax_ratios, lrw_lipreader_softmax_train_sorted_ratios[:, 0], lrw_lipreader_correct_or_wrong_train)
n_of_softmax_ratios_1_val, n_of_correct_in_softmax_ratios_1_val = eval(range_of_softmax_ratios, lrw_lipreader_softmax_val_sorted_ratios[:, 0], lrw_lipreader_correct_or_wrong_val)
n_of_softmax_ratios_1_test, n_of_correct_in_softmax_ratios_1_test = eval(range_of_softmax_ratios, lrw_lipreader_softmax_test_sorted_ratios[:, 0], lrw_lipreader_correct_or_wrong_test)
plot_samples_correct_predictions(range_of_softmax_ratios, n_of_softmax_ratios_1_train, n_of_correct_in_softmax_ratios_1_train, collect_type="train", x_label="Softmax Ratio 1")
plot_samples_correct_predictions(range_of_softmax_ratios, n_of_softmax_ratios_1_val, n_of_correct_in_softmax_ratios_1_val, collect_type="val", x_label="Softmax Ratio 1")
plot_samples_correct_predictions(range_of_softmax_ratios, n_of_softmax_ratios_1_test, n_of_correct_in_softmax_ratios_1_test, collect_type="test", x_label="Softmax Ratio 1")

# 2nd ratio
n_of_softmax_ratios_2_train, n_of_correct_in_softmax_ratios_2_train = eval(range_of_softmax_ratios, lrw_lipreader_softmax_train_sorted_ratios[:, 1], lrw_lipreader_correct_or_wrong_train)
n_of_softmax_ratios_2_val, n_of_correct_in_softmax_ratios_2_val = eval(range_of_softmax_ratios, lrw_lipreader_softmax_val_sorted_ratios[:, 1], lrw_lipreader_correct_or_wrong_val)
n_of_softmax_ratios_2_test, n_of_correct_in_softmax_ratios_2_test = eval(range_of_softmax_ratios, lrw_lipreader_softmax_test_sorted_ratios[:, 1], lrw_lipreader_correct_or_wrong_test)
plot_samples_correct_predictions(range_of_softmax_ratios, n_of_softmax_ratios_2_train, n_of_correct_in_softmax_ratios_2_train, collect_type="train", x_label="Softmax Ratio 2")
plot_samples_correct_predictions(range_of_softmax_ratios, n_of_softmax_ratios_2_val, n_of_correct_in_softmax_ratios_2_val, collect_type="val", x_label="Softmax Ratio 2")
plot_samples_correct_predictions(range_of_softmax_ratios, n_of_softmax_ratios_2_test, n_of_correct_in_softmax_ratios_2_test, collect_type="test", x_label="Softmax Ratio 2")

# 3rd ratio
n_of_softmax_ratios_3_train, n_of_correct_in_softmax_ratios_3_train = eval(range_of_softmax_ratios, lrw_lipreader_softmax_train_sorted_ratios[:, 3], lrw_lipreader_correct_or_wrong_train)
n_of_softmax_ratios_3_val, n_of_correct_in_softmax_ratios_3_val = eval(range_of_softmax_ratios, lrw_lipreader_softmax_val_sorted_ratios[:, 3], lrw_lipreader_correct_or_wrong_val)
n_of_softmax_ratios_3_test, n_of_correct_in_softmax_ratios_3_test = eval(range_of_softmax_ratios, lrw_lipreader_softmax_test_sorted_ratios[:, 3], lrw_lipreader_correct_or_wrong_test)
plot_samples_correct_predictions(range_of_softmax_ratios, n_of_softmax_ratios_3_val, n_of_correct_in_softmax_ratios_3_val, collect_type="val", x_label="Softmax Ratio 3")
plot_samples_correct_predictions(range_of_softmax_ratios, n_of_softmax_ratios_3_test, n_of_correct_in_softmax_ratios_3_test, collect_type="test", x_label="Softmax Ratio 3")

np.save('LRW_train_lipreader_softmax_ratios', lrw_lipreader_softmax_train_sorted_ratios[:, :2])
np.save('LRW_val_lipreader_softmax_ratios', lrw_lipreader_softmax_val_sorted_ratios[:, :2])
np.save('LRW_test_lipreader_softmax_ratios', lrw_lipreader_softmax_test_sorted_ratios[:, :2])


# LIPREADER PREDS vs WORD

lrw_train_correct_by_word_full = np.zeros((500, 200))
lrw_train_correct_by_word = np.zeros(500)
lrw_val_correct_by_word_full = np.zeros((500, 50))
lrw_val_correct_by_word = np.zeros(500)
lrw_test_correct_by_word_full = np.zeros((500, 50))
lrw_test_correct_by_word = np.zeros(500)
for i in range(500):
    lrw_train_correct_by_word[i] = np.mean(lrw_lipreader_correct_or_wrong_train[i:i+200])
    lrw_val_correct_by_word[i] = np.mean(lrw_lipreader_correct_or_wrong_val[i:i+50])
    lrw_test_correct_by_word[i] = np.mean(lrw_lipreader_correct_or_wrong_test[i:i+50])

plt.bar(np.arange(500), lrw_train_correct_by_word, label="LRW_train")
plt.bar(np.arange(500), lrw_val_correct_by_word, label="LRW_val", alpha=.7)
plt.bar(np.arange(500), lrw_test_correct_by_word, label="LRW_test", alpha=.7)
plt.yticks(np.arange(0, 1.1, .10))
plt.gca().yaxis.grid(True)
plt.legend()
# plt.xticks(np.arange(500), LRW_VOCAB, rotation='vertical', fontsize=6)
plt.title("Accuracy of lipreader on LRW")
plt.show()


# ABOUT
c = lrw_lipreader_correct_or_wrong_val
sm_true = lrw_lipreader_softmax_val_sorted_ratios[c==True]
sm_false = lrw_lipreader_softmax_val_sorted_ratios[c==False]

maxarg_to_correctarg = np.zeros(500)
for i, word in enumerate(LRW_VOCAB):
    maxarg_to_correctarg[lrw_correct_one_hot_y_arg[i*50]] = i

lipreader_preds_wordargs_val = maxarg_to_correctarg[np.argmax(lrw_lipreader_softmax_val, axis=1)]
correct_wordargs = maxarg_to_correctarg[lrw_correct_one_hot_y_arg]



