from assessor_functions import *

lrw_n_of_frames_per_sample_val = np.array(load_array_of_var_per_sample_from_csv(csv_file_name=N_OF_FRAMES_PER_SAMPLE_CSV_FILE, collect_type='val', collect_by='sample'))
lrw_n_of_frames_per_sample_test = np.array(load_array_of_var_per_sample_from_csv(csv_file_name=N_OF_FRAMES_PER_SAMPLE_CSV_FILE, collect_type='test', collect_by='sample'))

lrw_lipreader_dense_val, lrw_lipreader_softmax_val, lrw_correct_one_hot_y_arg = load_dense_softmax_y(collect_type='val')
lrw_lipreader_dense_test, lrw_lipreader_softmax_test, lrw_correct_one_hot_y_arg = load_dense_softmax_y(collect_type='test')

lrw_lipreader_correct_or_wrong_val = np.argmax(lrw_lipreader_softmax_val, axis=1) == lrw_correct_one_hot_y_arg
lrw_lipreader_correct_or_wrong_test = np.argmax(lrw_lipreader_softmax_test, axis=1) == lrw_correct_one_hot_y_arg

# LIPREADER PREDS vs N OF FRAMES

range_of_n_of_frames = np.arange(lrw_n_of_frames_per_sample_val.min(), lrw_n_of_frames_per_sample_val.max()+1)

def n_of_frames_eval(range_of_n_of_frames, lrw_n_of_frames_per_sample, lrw_lipreader_correct_or_wrong):
    num_of_samples = []
    num_of_correct_in_samples = []
    for n in range_of_n_of_frames:
        bool_of_samples_of_n_frames = lrw_n_of_frames_per_sample == n
        num_of_samples_of_n_frames = np.sum(bool_of_samples_of_n_frames)
        num_of_correct_in_samples_of_n_frames = np.sum(lrw_lipreader_correct_or_wrong[bool_of_samples_of_n_frames])
        num_of_samples.append(num_of_samples_of_n_frames)
        num_of_correct_in_samples.append(num_of_correct_in_samples_of_n_frames)
    return num_of_samples, num_of_correct_in_samples

num_of_samples_val, num_of_correct_in_samples_val = n_of_frames_eval(range_of_n_of_frames, lrw_n_of_frames_per_sample_val, lrw_lipreader_correct_or_wrong_val)
num_of_samples_test, num_of_correct_in_samples_test = n_of_frames_eval(range_of_n_of_frames, lrw_n_of_frames_per_sample_test, lrw_lipreader_correct_or_wrong_test)


def plot_samples_correct_predictions(range_of_n_of_frames, num_of_samples, num_of_correct_in_samples, collect_type="val"):
    percentage_of_correct_samples = np.array(num_of_correct_in_samples)/np.array(num_of_samples)*100
    plt.subplot(121)
    plt.bar(range_of_n_of_frames, num_of_samples, label="Total samples")
    plt.bar(range_of_n_of_frames, num_of_correct_in_samples, label="Lipreader correct predictions")
    plt.title("LRW " + collect_type + " total samples,\n# of correct predictions")
    plt.legend(fontsize=7)
    plt.xlabel("word duration")
    plt.subplot(122)
    plt.scatter(range_of_n_of_frames, percentage_of_correct_samples)
    plt.yticks(np.arange(0, 101, 10))
    plt.gca().yaxis.grid(True)
    plt.xlabel("(# of frames in sample)")
    plt.title("% of samples predicted\ncorrectly by lipreader")
    plt.show()

plot_samples_correct_predictions(range_of_n_of_frames, num_of_samples_val, num_of_correct_in_samples_val, collect_type="val")
plot_samples_correct_predictions(range_of_n_of_frames, num_of_samples_test, num_of_correct_in_samples_test, collect_type="test")


# LIPREADER PREDS vs BILABIALS

b_or_not = np.repeat(np.load('bilabial_or_not'), 50)

percentage_lipreader_correct_among_bilabials = np.mean(lrw_lipreader_correct_or_wrong_val[b_or_not == 1])*100
percentage_lipreader_correct_among_non_bilabials = np.mean(lrw_lipreader_correct_or_wrong_val[b_or_not == 0])*100
# >>> percentage_lipreader_correct_among_bilabials
# 74.76683937823833
# >>> percentage_lipreader_correct_among_non_bilabials
# 67.407166123778509

range_of_bilabials = np.arange(b_or_not.min(), b_or_not.max()+1)

n_of_b_or_not_val, n_of_correct_in_b_or_not_val = n_of_frames_eval(range_of_bilabials, b_or_not, lrw_lipreader_correct_or_wrong_val)
n_of_b_or_not_test, n_of_correct_in_b_or_not_test = n_of_frames_eval(range_of_bilabials, b_or_not, lrw_lipreader_correct_or_wrong_test)

plot_samples_correct_predictions(range_of_bilabials, n_of_b_or_not_val, n_of_correct_in_b_or_not_val, collect_type="val")
plot_samples_correct_predictions(range_of_bilabials, n_of_b_or_not_test, n_of_correct_in_b_or_not_test, collect_type="test")


# LIPREADER PREDS vs SYLLABLES

n_of_syllables = np.repeat(np.load('n_of_syllables'))

range_of_syllables = np.arange(n_of_syllables.min(), n_of_syllables.max()+1)

n_of_syllables_val, n_of_correct_in_syllables_val = n_of_frames_eval(range_of_syllables, n_of_syllables, lrw_lipreader_correct_or_wrong_val)
n_of_syllables_test, n_of_correct_in_syllables_test = n_of_frames_eval(range_of_syllables, n_of_syllables, lrw_lipreader_correct_or_wrong_test)

plot_samples_correct_predictions(range_of_syllables, n_of_syllables_val, n_of_correct_in_syllables_val, collect_type="val")
plot_samples_correct_predictions(range_of_syllables, n_of_syllables_test, n_of_correct_in_syllables_test, collect_type="test")

