"""
朴素贝叶斯文本情感分析 - 独立性假设失效分析
依赖包：numpy, pandas, scikit-learn, matplotlib
"""

import warnings
warnings.filterwarnings('ignore', message=".*glyph.*")

import numpy as np
import pandas as pd
from collections import defaultdict
import re
import os
import sys
import matplotlib.pyplot as plt
import matplotlib as mpl
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial']
plt.rcParams['axes.unicode_minus'] = False

# 路径设置
base_dir = os.path.dirname(__file__)
data_path = os.path.join(base_dir, 'aclImdb')

# ==================== 可视化模块 ====================

def plot_convergence_chart(llm_results, output_path=None):
    """绘制大数定律收敛图"""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    words = list(llm_results.keys())[:2]

    ax1 = axes[0]
    for word in words:
        data = llm_results[word]
        ax1.plot(data['sizes'], data['probs'], 'o-', label=f"'{word}'", markersize=6)
    ax1.set_xlabel('Sample Size', fontsize=12)
    ax1.set_ylabel('P(word|positive)', fontsize=12)
    ax1.set_title('Law of Large Numbers: Probability Estimation', fontsize=14)
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)

    ax2 = axes[1]
    for word in words:
        data = llm_results[word]
        # 用plot代替semilogy避免Unicode minus符号问题
        valid_stds = [s if s > 0 else 1e-10 for s in data['stds']]
        ax2.plot(data['sizes'], valid_stds, 's-', label=f"'{word}'", markersize=6)
        ax2.set_yscale('log')
    ax2.set_xlabel('Sample Size', fontsize=12)
    ax2.set_ylabel('Standard Deviation (log)', fontsize=12)
    ax2.set_title('Estimation Stability: Std Dev Decreases', fontsize=14)
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Figure saved: {output_path}")
    plt.close()


def plot_experiment_comparison(results, output_path=None):
    """绘制对比实验结果柱状图"""
    fig, ax = plt.subplots(figsize=(10, 6))
    methods = list(results.keys())
    metrics = ['accuracy', 'precision', 'recall', 'f1']
    x = np.arange(len(methods))
    width = 0.2

    for i, metric in enumerate(metrics):
        values = [results[m][metric] for m in methods]
        bars = ax.bar(x + i * width, values, width, label=metric.capitalize())
        for bar, val in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                   f'{val:.2%}', ha='center', va='bottom', fontsize=8)

    ax.set_ylabel('Score', fontsize=12)
    ax.set_title('Experiment Comparison: Three Methods', fontsize=14)
    ax.set_xticks(x + width * 1.5)
    ax.set_xticklabels(['Unigram', 'Bigram', 'Unigram\n(Filtered)'])
    ax.legend(loc='lower right')
    ax.set_ylim(0.75, 0.90)
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Figure saved: {output_path}")
    plt.close()


def plot_correlation_distribution(report, output_path=None):
    """绘制相关系数分布直方图"""
    corr_values = [corr for _, _, corr in report['strong_corr_words']]
    if not corr_values:
        print("No correlation data to plot")
        return

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    ax1 = axes[0]
    ax1.hist(corr_values, bins=20, edgecolor='black', alpha=0.7, color='steelblue')
    ax1.set_xlabel('Correlation Coefficient', fontsize=12)
    ax1.set_ylabel('Frequency', fontsize=12)
    ax1.set_title('Distribution of Strongly Correlated Word Pairs', fontsize=14)
    ax1.grid(True, alpha=0.3, axis='y')
    mean_corr = np.mean(corr_values)
    ax1.axvline(mean_corr, color='red', linestyle='--', linewidth=2, label=f'Mean = {mean_corr:.3f}')
    ax1.legend()

    ax2 = axes[1]
    word_pairs = [f"{w1}\n{w2}" for w1, w2, _ in report['strong_corr_words'][:10]]
    corrs = [corr for _, _, corr in report['strong_corr_words'][:10]]
    bars = ax2.barh(range(len(word_pairs)), corrs, color='coral', edgecolor='black')
    ax2.set_yticks(range(len(word_pairs)))
    ax2.set_yticklabels(word_pairs, fontsize=9)
    ax2.set_xlabel('Correlation Coefficient', fontsize=12)
    ax2.set_title('Top 10 Strongly Correlated Word Pairs', fontsize=14)
    ax2.invert_yaxis()
    ax2.grid(True, alpha=0.3, axis='x')
    for bar, corr in zip(bars, corrs):
        ax2.text(bar.get_width() + 0.02, bar.get_y() + bar.get_height()/2,
                f'{corr:.3f}', va='center', fontsize=9)

    plt.tight_layout()
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Figure saved: {output_path}")
    plt.close()


def save_all_figures(report, results, llm_results, output_dir=None):
    """保存所有图表"""
    plot_convergence_chart(llm_results, os.path.join(output_dir, "fig1_convergence.png"))
    plot_experiment_comparison(results, os.path.join(output_dir, "fig2_comparison.png"))
    plot_correlation_distribution(report, os.path.join(output_dir, "fig3_correlation.png"))
    print("All figures saved!")


# ==================== 数据预处理模块 ====================

def load_imdb_data(data_dir, max_samples=None):
    """加载IMDB数据集"""
    texts, labels = [], []
    if not (os.path.exists(os.path.join(data_dir, 'pos')) and os.path.exists(os.path.join(data_dir, 'neg'))):
        train_dir = os.path.join(data_dir, 'train')
        if os.path.exists(train_dir):
            data_dir = train_dir

    for label in ['pos', 'neg']:
        folder = os.path.join(data_dir, label)
        if not os.path.exists(folder):
            print("警告: 未找到 " + folder)
            continue
        files = [f for f in os.listdir(folder) if f.endswith('.txt')]
        if max_samples is not None:
            files = files[:max_samples]
        for filename in files:
            filepath = os.path.join(folder, filename)
            try:
                with open(filepath, 'r', encoding='utf-8') as f:
                    texts.append(f.read())
                labels.append(1 if label == 'pos' else 0)
            except Exception as e:
                print("读取失败 " + filepath + ": " + str(e))

    print("加载完成: 正面" + str(sum(labels)) + "条, 负面" + str(len(labels)-sum(labels)) + "条")
    return texts, labels


STOP_WORDS = {
    'the', 'a', 'an', 'and', 'or', 'but', 'is', 'are', 'was', 'were',
    'be', 'been', 'being', 'have', 'has', 'had', 'do', 'does', 'did',
    'will', 'would', 'could', 'should', 'may', 'might', 'must', 'can',
    'to', 'of', 'in', 'for', 'on', 'with', 'at', 'by', 'from', 'as',
    'it', 'this', 'that', 'these', 'those', 'i', 'you', 'he', 'she',
    'we', 'they', 'what', 'which', 'who', 'whom', 'whose', 'where',
    'when', 'why', 'how', 'not', 'no', 'yes', 'all', 'any', 'some',
    'there', 'here', 'very', 'just', 'also', 'so', 'than', 'too'
}

def simple_tokenize(text):
    """简单分词：去HTML标签、转小写、过滤停用词和短词"""
    text = re.sub(r'<[^>]+>', ' ', text.lower())
    text = re.sub(r'[^a-z\s]', ' ', text)
    words = text.split()
    return [w for w in words if len(w) > 1 and w not in STOP_WORDS]


def build_vocabulary(texts, min_freq=5, max_vocab=5000):
    """构建词表：统计词频、过滤低频词、取高频词"""
    word_freq = defaultdict(int)
    for text in texts:
        for w in simple_tokenize(text):
            word_freq[w] += 1
    word_freq = {w: f for w, f in word_freq.items() if f >= min_freq}
    sorted_words = sorted(word_freq.items(), key=lambda x: -x[1])[:max_vocab]
    return {w: i for i, (w, _) in enumerate(sorted_words)}, word_freq


def text_to_bow(text, word2idx):
    """文本转词袋向量"""
    words = simple_tokenize(text)
    bow = np.zeros(len(word2idx))
    for w in words:
        if w in word2idx:
            bow[word2idx[w]] += 1
    return bow


def text_to_bigram_bow(text, bigram_word2idx, word2idx):
    """文本转bigram词袋向量"""
    words = simple_tokenize(text)
    bow = np.zeros(len(bigram_word2idx))
    for w in words:
        if w in bigram_word2idx:
            bow[bigram_word2idx[w]] += 1
    for i in range(len(words) - 1):
        bigram = words[i] + "_" + words[i+1]
        if bigram in bigram_word2idx:
            bow[bigram_word2idx[bigram]] += 1
    return bow


# ==================== 朴素贝叶斯分类器 ====================

class NaiveBayesClassifier:
    """朴素贝叶斯分类器（伯努利模型 + 拉普拉斯平滑）"""

    def __init__(self, alpha=1.0):
        self.alpha = alpha

    def fit(self, X, y, feature_names=None):
        """训练模型：计算P(类别)和P(词|类别)"""
        n_samples, self.vocab_size = X.shape
        self.feature_names = feature_names or list(range(self.vocab_size))
        n_positive = np.sum(y == 1)
        n_negative = np.sum(y == 0)
        self.class_probs = {1: n_positive / n_samples, 0: n_negative / n_samples}
        self.word_probs = {}
        for c in [0, 1]:
            X_c = X[y == c]
            word_counts = np.sum(X_c, axis=0) + self.alpha
            total_words = np.sum(X_c) + self.alpha * self.vocab_size
            self.word_probs[c] = word_counts / total_words
        return self

    def predict_proba(self, X):
        """预测概率（使用对数避免下溢）"""
        n_samples = X.shape[0]
        log_proba = np.zeros((n_samples, 2))
        for c in [0, 1]:
            log_proba[:, c] = np.log(self.class_probs[c]) + X @ np.log(self.word_probs[c])
        log_proba_max = np.max(log_proba, axis=1, keepdims=True)
        log_proba -= log_proba_max
        proba = np.exp(log_proba)
        proba /= np.sum(proba, axis=1, keepdims=True)
        return proba

    def predict(self, X):
        """预测类别"""
        proba = self.predict_proba(X)
        return (proba[:, 1] > 0.5).astype(int)


# ==================== 独立性分析模块 ====================

def compute_word_covariance(X, y):
    """计算词与词之间的相关系数矩阵"""
    X_pos = X[y == 1]
    std_dev = np.std(X_pos, axis=0)
    valid_mask = std_dev > 0
    X_pos_filtered = X_pos[:, valid_mask]

    if X_pos_filtered.shape[1] == 0:
        return np.array([]), np.array([]), []

    cov_matrix = np.cov(X_pos_filtered, rowvar=False)
    corr_matrix = np.corrcoef(X_pos_filtered, rowvar=False)
    valid_indices = np.where(valid_mask)[0]

    pos_corr_words = []
    for i in range(X_pos_filtered.shape[1]):
        for j in range(i + 1, X_pos_filtered.shape[1]):
            if not np.isnan(corr_matrix[i, j]) and corr_matrix[i, j] > 0.3:
                pos_corr_words.append((valid_indices[i], valid_indices[j], corr_matrix[i, j]))
    pos_corr_words.sort(key=lambda x: -x[2])

    return cov_matrix, corr_matrix, pos_corr_words


def analyze_independence_violation(X, y, feature_names):
    """分析独立性假设失效程度"""
    _, corr_matrix, strong_corr = compute_word_covariance(X, y)
    valid_corr = corr_matrix[~np.isnan(corr_matrix)]
    valid_corr = valid_corr[valid_corr != 1.0]

    word_list = list(feature_names) if feature_names else []
    strong_corr_words_named = []
    for i, j, corr in strong_corr[:20]:
        if i < len(word_list) and j < len(word_list):
            strong_corr_words_named.append((word_list[i], word_list[j], corr))

    report = {
        'mean_abs_corr': np.mean(np.abs(valid_corr)),
        'max_corr': np.max(valid_corr),
        'strong_corr_count': len(strong_corr),
        'strong_corr_words': strong_corr_words_named,
        'corr_above_02': np.sum(valid_corr > 0.2) / len(valid_corr) * 100,
        'corr_above_05': np.sum(valid_corr > 0.5) / len(valid_corr) * 100,
    }
    return report


# ==================== 对比实验模块 ====================

def run_comparison_experiment(texts, labels, word2idx):
    """运行三组对比实验：单词特征、Bigram特征、去除高共现词"""
    from sklearn.model_selection import train_test_split

    results = {}
    X = np.array([text_to_bow(t, word2idx) for t in texts])
    y = np.array(labels)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # 实验1: 单词特征
    nb1 = NaiveBayesClassifier(alpha=1.0)
    nb1.fit(X_train, y_train, feature_names=list(word2idx.keys()))
    y_pred = nb1.predict(X_test)
    results['unigram'] = {
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred),
        'recall': recall_score(y_test, y_pred),
        'f1': f1_score(y_test, y_pred),
    }

    # 实验2: Bigram特征
    all_bigrams = defaultdict(int)
    for text in texts:
        words = simple_tokenize(text)
        for i in range(len(words) - 1):
            all_bigrams[words[i] + "_" + words[i+1]] += 1
    top_bigrams = [bg for bg, _ in sorted(all_bigrams.items(), key=lambda x: -x[1])[:1000]]
    bigram_word2idx = word2idx.copy()
    for bg in top_bigrams:
        if bg not in bigram_word2idx:
            bigram_word2idx[bg] = len(bigram_word2idx)

    X2 = np.array([text_to_bigram_bow(t, bigram_word2idx, word2idx) for t in texts])
    X2_train, X2_test, y2_train, y2_test = train_test_split(X2, y, test_size=0.2, random_state=42)
    nb2 = NaiveBayesClassifier(alpha=1.0)
    nb2.fit(X2_train, y2_train, feature_names=list(bigram_word2idx.keys()))
    y2_pred = nb2.predict(X2_test)
    results['bigram'] = {
        'accuracy': accuracy_score(y2_test, y2_pred),
        'precision': precision_score(y2_test, y2_pred),
        'recall': recall_score(y2_test, y2_pred),
        'f1': f1_score(y2_test, y2_pred),
    }

    # 实验3: 去除高共现词
    _, _, strong_corr = compute_word_covariance(X, y)
    high_corr_indices = set()
    for i, j, _ in strong_corr[:50]:
        high_corr_indices.add(i)
        high_corr_indices.add(j)
    new_word2idx = {}
    idx = 0
    for w, i in word2idx.items():
        if i not in high_corr_indices:
            new_word2idx[w] = idx
            idx += 1

    X3 = np.array([text_to_bow(t, new_word2idx) for t in texts])
    X3_train, X3_test, y3_train, y3_test = train_test_split(X3, y, test_size=0.2, random_state=42)
    nb3 = NaiveBayesClassifier(alpha=1.0)
    nb3.fit(X3_train, y3_train, feature_names=list(new_word2idx.keys()))
    y3_pred = nb3.predict(X3_test)
    results['unigram_filtered'] = {
        'accuracy': accuracy_score(y3_test, y3_pred),
        'precision': precision_score(y3_test, y3_pred),
        'recall': recall_score(y3_test, y3_pred),
        'f1': f1_score(y3_test, y3_pred),
    }

    return results


# ==================== 大数定律验证模块 ====================

def verify_law_of_large_numbers(X, y, word2idx, target_words=None):
    """验证大数定律：样本量增加时条件概率估计趋于稳定"""
    if target_words is None:
        target_words = list(word2idx.keys())[:5]
    X_pos = X[y == 1]
    sample_sizes = [100, 500, 1000, 2000, 5000, 10000, 20000]
    sample_sizes = sorted(set([min(s, len(X_pos)) for s in sample_sizes]))

    results = {word: {'sizes': [], 'probs': [], 'stds': []} for word in target_words}

    for size in sample_sizes:
        n_trials = 20
        prob_estimates = {word: [] for word in target_words}
        for _ in range(n_trials):
            indices = np.random.choice(len(X_pos), size=size, replace=False)
            X_sample = X_pos[indices]
            word_probs = np.mean(X_sample > 0, axis=0)
            for word in target_words:
                if word in word2idx:
                    prob_estimates[word].append(word_probs[word2idx[word]])
        for word in target_words:
            probs = prob_estimates[word]
            if probs:
                results[word]['sizes'].append(size)
                results[word]['probs'].append(np.mean(probs))
                results[word]['stds'].append(np.std(probs))

    return results


# ==================== 结果保存 ====================

def save_results_to_file(report, results, llm_results, word2idx, output_file=None):
    """将结果保存到文件"""
    if output_file is None:
        output_file = os.path.join(base_dir, '实验结果.txt')

    with open(output_file, 'w', encoding='utf-8') as f:
        f.write("=" * 60 + "\n")
        f.write("朴素贝叶斯独立性假设失效分析 - 实验结果\n")
        f.write("=" * 60 + "\n\n")

        f.write("1. 独立性假设失效分析\n")
        f.write("-" * 40 + "\n")
        f.write(f"平均绝对相关系数: {report['mean_abs_corr']:.4f}\n")
        f.write(f"最大相关系数: {report['max_corr']:.4f}\n")
        f.write(f"强相关词对数量 (>0.3): {report['strong_corr_count']}\n")
        f.write(f"相关系数>0.2的比例: {report['corr_above_02']:.1f}%\n")
        f.write(f"相关系数>0.5的比例: {report['corr_above_05']:.1f}%\n\n")

        f.write("前10个强相关词对:\n")
        for i, (w1, w2, corr) in enumerate(report['strong_corr_words'][:10]):
            f.write(f"  {i+1}. {w1} & {w2}: r={corr:.4f}\n")
        f.write("\n")

        f.write("2. 对比实验结果\n")
        f.write("-" * 40 + "\n")
        f.write(f"{'方法':<20} {'准确率':>10} {'精确率':>10} {'召回率':>10} {'F1值':>10}\n")
        for method, metrics in results.items():
            f.write(f"{method:<20} {metrics['accuracy']:>10.4f} {metrics['precision']:>10.4f} "
                   f"{metrics['recall']:>10.4f} {metrics['f1']:>10.4f}\n")
        f.write("\n")

        f.write("3. 大数定律验证结果\n")
        f.write("-" * 40 + "\n")
        for word, data in llm_results.items():
            if data['probs']:
                f.write(f"\n词 '{word}':\n")
                for size, prob, std in zip(data['sizes'], data['probs'], data['stds']):
                    f.write(f"  样本量={size}: P={prob:.4f}, 标准差={std:.4f}\n")

    print(f"\n结果已保存到: {output_file}")


def load_imdb_full():
    """加载IMDB完整数据集（训练集和测试集分开）"""
    train_texts, train_labels = load_imdb_data(os.path.join(data_path, 'train'), max_samples=None)
    test_texts, test_labels = load_imdb_data(os.path.join(data_path, 'test'), max_samples=None)
    print("训练集: " + str(len(train_texts)) + "条")
    print("测试集: " + str(len(test_texts)) + "条")
    return train_texts, train_labels, test_texts, test_labels


def main():
    """主程序：一键运行完整流程"""
    print("=" * 60)
    print("朴素贝叶斯文本情感分析 - 独立性假设失效分析")
    print("=" * 60)

    # 步骤1: 加载完整数据集
    print("\n[1/5] 加载IMDB数据集...")
    train_texts, train_labels, test_texts, test_labels = load_imdb_full()

    # 步骤2: 构建词表（只用训练集）
    print("\n[2/5] 构建词表...")
    word2idx, word_freq = build_vocabulary(train_texts, min_freq=5, max_vocab=5000)
    print("词表大小: " + str(len(word2idx)))

    # 步骤3: 独立性假设失效分析（用训练集）
    print("\n[3/5] 独立性假设失效分析...")
    X_train = np.array([text_to_bow(t, word2idx) for t in train_texts])
    y_train = np.array(train_labels)
    report = analyze_independence_violation(X_train, y_train, list(word2idx.keys()))

    print("\n平均绝对相关系数: " + str(round(report['mean_abs_corr'], 4)))
    print("最大相关系数: " + str(round(report['max_corr'], 4)))
    print("强相关词对数量 (>0.3): " + str(report['strong_corr_count']))
    print("相关系数>0.2的比例: " + str(round(report['corr_above_02'], 1)) + "%")
    print("相关系数>0.5的比例: " + str(round(report['corr_above_05'], 1)) + "%")

    print("\n前10个强相关词对:")
    for i, (w1, w2, corr) in enumerate(report['strong_corr_words'][:10]):
        print("  " + str(w1) + " & " + str(w2) + ": r=" + str(round(corr, 4)))

    # 步骤4: 对比实验
    print("\n[4/5] 运行对比实验...")
    from sklearn.model_selection import train_test_split
    results = {}

    # 实验1: 单词特征
    nb1 = NaiveBayesClassifier(alpha=1.0)
    nb1.fit(X_train, y_train, feature_names=list(word2idx.keys()))
    X_test = np.array([text_to_bow(t, word2idx) for t in test_texts])
    y_test = np.array(test_labels)
    y1_pred = nb1.predict(X_test)
    results['unigram'] = {
        'accuracy': accuracy_score(y_test, y1_pred),
        'precision': precision_score(y_test, y1_pred),
        'recall': recall_score(y_test, y1_pred),
        'f1': f1_score(y_test, y1_pred),
    }

    # 实验2: Bigram特征
    all_bigrams = defaultdict(int)
    for text in train_texts:
        words = simple_tokenize(text)
        for i in range(len(words) - 1):
            all_bigrams[words[i] + "_" + words[i+1]] += 1
    top_bigrams = [bg for bg, _ in sorted(all_bigrams.items(), key=lambda x: -x[1])[:1000]]
    bigram_word2idx = word2idx.copy()
    for bg in top_bigrams:
        if bg not in bigram_word2idx:
            bigram_word2idx[bg] = len(bigram_word2idx)

    X2_train = np.array([text_to_bigram_bow(t, bigram_word2idx, word2idx) for t in train_texts])
    X2_test = np.array([text_to_bigram_bow(t, bigram_word2idx, word2idx) for t in test_texts])
    nb2 = NaiveBayesClassifier(alpha=1.0)
    nb2.fit(X2_train, y_train, feature_names=list(bigram_word2idx.keys()))
    y2_pred = nb2.predict(X2_test)
    results['bigram'] = {
        'accuracy': accuracy_score(y_test, y2_pred),
        'precision': precision_score(y_test, y2_pred),
        'recall': recall_score(y_test, y2_pred),
        'f1': f1_score(y_test, y2_pred),
    }

    # 实验3: 去除高共现词
    _, _, strong_corr = compute_word_covariance(X_train, y_train)
    high_corr_indices = set()
    for i, j, _ in strong_corr[:50]:
        high_corr_indices.add(i)
        high_corr_indices.add(j)
    new_word2idx = {}
    idx = 0
    for w, i in word2idx.items():
        if i not in high_corr_indices:
            new_word2idx[w] = idx
            idx += 1

    X3_train = np.array([text_to_bow(t, new_word2idx) for t in train_texts])
    X3_test = np.array([text_to_bow(t, new_word2idx) for t in test_texts])
    nb3 = NaiveBayesClassifier(alpha=1.0)
    nb3.fit(X3_train, y_train, feature_names=list(new_word2idx.keys()))
    y3_pred = nb3.predict(X3_test)
    results['unigram_filtered'] = {
        'accuracy': accuracy_score(y_test, y3_pred),
        'precision': precision_score(y_test, y3_pred),
        'recall': recall_score(y_test, y3_pred),
        'f1': f1_score(y_test, y3_pred),
    }

    # 打印实验结果对比
    print("\n实验结果对比:")
    print("-" * 60)
    print(f"{'方法':<20} {'准确率':>10} {'精确率':>10} {'召回率':>10} {'F1值':>10}")
    print("-" * 60)
    for method, metrics in results.items():
        print(f"{method:<20} {metrics['accuracy']:>10.4f} {metrics['precision']:>10.4f} "
              f"{metrics['recall']:>10.4f} {metrics['f1']:>10.4f}")
    print("-" * 60)

    # 步骤5: 大数定律验证
    print("\n[5/5] 大数定律验证...")
    target_words = list(word2idx.keys())[:5]
    llm_results = verify_law_of_large_numbers(X_train, y_train, word2idx, target_words)

    print("\n条件概率收敛情况:")
    for word, data in llm_results.items():
        if data['probs']:
            print(f"\n词 '{word}':")
            for size, prob, std in zip(data['sizes'], data['probs'], data['stds']):
                print(f"  样本量={size}: P={prob:.4f}, 标准差={std:.4f}")

    # 保存结果
    print("\n" + "=" * 60)
    print("分析完成!")
    print("=" * 60)

    save_results_to_file(report, results, llm_results, word2idx)
    save_all_figures(report, results, llm_results, output_dir=base_dir)
    print("图表已保存!")

    return results, report, llm_results, word2idx


if __name__ == "__main__":
    results, report, llm_results, word2idx = main()
