import pandas as pd
import numpy as np
import re
import ast
from collections import Counter, defaultdict
from scipy.stats import chi2
import matplotlib.pyplot as plt

# ============================================================================
# CONFIGURATION
# ============================================================================

DATA_PATH ='data/processed/2025-11-20/processed_dataset.csv'

# Define technical vocabulary in BOTH languages
TECH_WORDS_EN = {
    'api', 'deploy', 'deployment', 'database', 'feature', 'latency', 'cache',
    'scale', 'scalability', 'optimize', 'optimization', 'performance', 'code',
    'build', 'test', 'debug', 'branch', 'commit', 'pull', 'request',
    'merge', 'refactor', 'architecture', 'framework', 'library', 'module',
    'function', 'class', 'variable', 'algorithm', 'complexity', 'buffer',
    'queue', 'stack', 'tree', 'graph', 'async', 'thread', 'process', 'server',
    'client', 'cloud', 'docker', 'kubernetes', 'microservice', 'pipeline',
    'monitoring', 'logging', 'metric', 'alert', 'failover', 'redundancy',
    'throughput', 'bandwidth', 'protocol', 'encryption', 'authentication',
    'authorization', 'session', 'token', 'endpoint', 'interface', 'middleware',
    'pr', 'staging', 'qa', 'bug', 'deadline', 'sprint', 'release', 'production',
}

# Mandarin technical vocabulary (Chinese characters)
TECH_WORDS_ZH = {
    '接口',      # API/interface
    '部署',      # deploy
    '数据库',    # database
    '功能',      # feature
    '延迟',      # latency
    '缓存',      # cache
    '扩展',      # scale
    '扩展性',    # scalability
    '优化',      # optimize
    '性能',      # performance
    '代码',      # code
    '构建',      # build
    '测试',      # test
    '调试',      # debug
    '分支',      # branch
    '提交',      # commit
    '拉取',      # pull
    '合并',      # merge
    '重构',      # refactor
    '架构',      # architecture
    '框架',      # framework
    '库',        # library
    '模块',      # module
    '函数',      # function
    '类',        # class
    '变量',      # variable
    '算法',      # algorithm
    '复杂度',    # complexity
    '缓冲',      # buffer
    '队列',      # queue
    '栈',        # stack
    '树',        # tree
    '图',        # graph
    '异步',      # async
    '线程',      # thread
    '进程',      # process
    '服务器',    # server
    '客户端',    # client
    '云',        # cloud
    '容器',      # container (docker-related)
    '微服务',    # microservice
    '管道',      # pipeline
    '监控',      # monitoring
    '日志',      # logging
    '指标',      # metric
    '告警',      # alert
    '冗余',      # redundancy
    '吞吐量',    # throughput
    '带宽',      # bandwidth
    '协议',      # protocol
    '加密',      # encryption
    '认证',      # authentication
    '授权',      # authorization
    '会话',      # session
    '令牌',      # token
    '端点',      # endpoint
    '中间件',    # middleware
    '问题',      # issue/bug
    '截止日期',  # deadline
    '发布',      # release
    '生产',      # production
    '质量保证',   # QA
}

# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def parse_labels(x):
    """Parse token-level language labels from CSV"""
    if pd.isna(x):
        return []
    if isinstance(x, list):
        return x
    s = str(x).strip()
    try:
        v = ast.literal_eval(s)
        if isinstance(v, list):
            return v
    except Exception:
        pass
    if ',' in s:
        return [p.strip() for p in s.split(',') if p.strip()]
    return [s]

def has_english(text):
    """Check if text contains English characters"""
    return bool(re.search(r'[A-Za-z]', str(text)))

def has_mandarin(text):
    """Check if text contains Mandarin characters"""
    return bool(re.search(r'[\u4e00-\u9fff]', str(text)))

def is_code_switched(text):
    """Check if text contains BOTH English and Mandarin"""
    return has_english(text) and has_mandarin(text)

def count_technical_terms_en(text):
    """Count English technical terms in text"""
    text_lower = str(text).lower()
    count = 0
    for term in TECH_WORDS_EN:
        # Use word boundaries to avoid partial matches
        pattern = r'\b' + re.escape(term) + r'\b'
        count += len(re.findall(pattern, text_lower))
    return count

def count_technical_terms_zh(text):
    """Count Mandarin technical terms in text"""
    text_str = str(text)
    count = 0
    for term in TECH_WORDS_ZH:
        count += text_str.count(term)
    return count

def classify_technical_preference(text):
    """Classify whether English or Mandarin technical terms dominate"""
    en_count = count_technical_terms_en(text)
    zh_count = count_technical_terms_zh(text)
    
    if en_count > zh_count:
        return 'ENGLISH'
    elif zh_count > en_count:
        return 'MANDARIN'
    elif en_count == zh_count and en_count > 0:
        return 'MIXED'
    else:
        return 'NO_TECH'

def extract_language_from_labels(labels_list):
    """Extract primary language from token-level labels"""
    if not labels_list:
        return 'UNK'
    
    lang_counts = Counter(labels_list)
    # Remove punctuation and 'other'
    filtered = {k: v for k, v in lang_counts.items() if k not in ['punct', 'other']}
    
    if not filtered:
        return 'UNK'
    
    # Determine if truly mixed (both en and zh present)
    has_en = 'en' in filtered
    has_zh = 'zh' in filtered
    
    if has_en and has_zh:
        return 'mix'
    elif has_zh:
        return 'zh'
    elif has_en:
        return 'en'
    else:
        return 'UNK'

# ============================================================================
# MAIN ANALYSIS
# ============================================================================

def analyze_authority_skew(data_path):
    """
    Calculate Technical Jargon Authority Index (TJAI) and related metrics
    """
    
    print("MANDARIN-ENGLISH AUTHORITY SKEW ANALYSIS")
    
    # Load dataset
    print("[1/7] Loading dataset...")
    df = pd.read_csv(data_path, encoding='utf-8')
    print(f"     Loaded {len(df)} turns")
    print(f"     Unique conversations: {df['id'].nunique()}")
    
    # Parse labels column for token-level language info
    print("[2/7] Parsing token-level language labels...")
    df['labels_parsed'] = df['labels'].apply(parse_labels)
    
    # Extract primary language for each turn
    print("[3/7] Extracting primary language for each turn...")
    df['primary_lang'] = df['labels_parsed'].apply(extract_language_from_labels)
    
    print(f"     Language distribution:")
    print(f"       English-only:  {(df['primary_lang'] == 'en').sum()} turns")
    print(f"       Mandarin-only: {(df['primary_lang'] == 'zh').sum()} turns")
    print(f"       Mixed:         {(df['primary_lang'] == 'mix').sum()} turns")
    print(f"       Unknown:       {(df['primary_lang'] == 'UNK').sum()} turns")
    
    # Filter to code-switched turns (manual check + language labels)
    print("[4/7] Filtering to code-switched turns...")
    df['is_code_switched'] = df['utterance'].apply(is_code_switched)
    df_code_switched = df[df['is_code_switched']].copy()
    print(f"     Found {len(df_code_switched)} code-switched turns")
    print(f"     Percentage: {len(df_code_switched)/len(df)*100:.1f}%")
    
    # Skip the mixed-language filtering from labels (it's not parsing)
    # Instead, just use the code-switched turns we found
    print("[5/7] Using code-switched turns directly (skipping label-based filtering)...")
    print(f"     Working with {len(df_code_switched)} code-switched turns")
    
    # Count technical terms
    print("[6/7] Counting technical terms (English vs Mandarin)...")
    df_code_switched['tech_count_en'] = df_code_switched['utterance'].apply(
        count_technical_terms_en
    )
    df_code_switched['tech_count_zh'] = df_code_switched['utterance'].apply(
        count_technical_terms_zh
    )
    df_code_switched['has_technical'] = (
        (df_code_switched['tech_count_en'] > 0) | 
        (df_code_switched['tech_count_zh'] > 0)
    )
    
    # Filter to turns with at least one technical term
    df_technical = df_code_switched[df_code_switched['has_technical']].copy()
    print(f"     Found {len(df_technical)} code-switched turns with technical content")
    
    # Classify technical preference
    print("[7/7] Classifying technical term preference...")
    df_technical['tech_preference'] = df_technical['utterance'].apply(
        classify_technical_preference
    )
    
    
    preference_counts = df_technical['tech_preference'].value_counts()
    english_count = preference_counts.get('ENGLISH', 0)
    mandarin_count = preference_counts.get('MANDARIN', 0)
    mixed_count = preference_counts.get('MIXED', 0)
    no_tech_count = preference_counts.get('NO_TECH', 0)
    
    total_technical = english_count + mandarin_count + mixed_count
    
    # Calculate TJAI (excluding MIXED from denominator)
    tjai_denominator = english_count + mandarin_count
    if tjai_denominator > 0:
        tjai = english_count / tjai_denominator
    else:
        tjai = 0.0
    
    authority_ratio = english_count / mandarin_count if mandarin_count > 0 else 0
    
    # ========================================================================
    # CHI-SQUARE TEST
    # ========================================================================
    
    if tjai_denominator > 0:
        observed = [english_count, mandarin_count]
        expected = [tjai_denominator * 0.5, tjai_denominator * 0.5]
        
        # Avoid division by zero
        chi2_stat = sum((o - e)**2 / e for o, e in zip(observed, expected) if e > 0)
        p_value = 1 - chi2.cdf(chi2_stat, df=1)
        
        # Effect size: Cramér's V
        n = tjai_denominator
        cramers_v = np.sqrt(chi2_stat / n) if n > 0 else 0
    else:
        chi2_stat = 0.0
        p_value = 1.0
        cramers_v = 0.0

    
    print("=" * 80)
    print("RESULTS: TECHNICAL JARGON AUTHORITY INDEX (TJAI)")
    
    print("PRIMARY FINDINGS:")
    print(f"  English technical terms preferred:  {english_count:,} turns ({english_count/tjai_denominator*100:.1f}%)")
    print(f"  Mandarin technical terms preferred: {mandarin_count:,} turns ({mandarin_count/tjai_denominator*100:.1f}%)")
    print(f"  Mixed preference (equal):           {mixed_count:,} turns")
    print(f"  Total code-switched technical:      {tjai_denominator:,} turns")
    
    print("AUTHORITY SKEW METRICS:")
    print(f"  TJAI (English %):                   {tjai*100:.1f}%")
    print(f"  Baseline (balanced):                50.0%")
    print(f"  Skew magnitude:                     {(tjai - 0.5)*100:+.1f} percentage points")
    print(f"  Authority ratio (EN/ZH):            {authority_ratio:.2f}x")
    
    print("STATISTICAL SIGNIFICANCE:")
    print(f"  Chi-square statistic:               χ² = {chi2_stat:.2f}")
    print(f"  P-value:                            p = {p_value:.6f}")
    print(f"  Effect size (Cramér's V):           V = {cramers_v:.3f}")
    if p_value < 0.001:
        print(f"  Significance level:                 *** HIGHLY SIGNIFICANT (p < 0.001)")
    elif p_value < 0.05:
        print(f"  Significance level:                 ** SIGNIFICANT (p < 0.05)")
    else:
        print(f"  Significance level:                 NOT SIGNIFICANT (p ≥ 0.05)")

    
    print("DOMAIN-SPECIFIC ANALYSIS")
    
    domains = {
        'PR/Merge': ['pr', 'merge', '合并'],
        'Deploy': ['deploy', 'deployment', '部署'],
        'Feature': ['feature', '功能'],
        'Database': ['database', '数据库'],
        'Build/Test': ['build', 'test', 'bug', 'qa', '构建', '测试', '调试'],
        'Performance': ['performance', 'latency', 'optimize', '性能', '延迟', '优化'],
        'Cache': ['cache', '缓存'],
        'Staging/Production': ['staging', 'production', '生产', '上线'],
    }
    
    domain_results = {}
    
    for domain_name, keywords in domains.items():
        mask = df_technical['utterance'].apply(
            lambda x: any(kw in str(x).lower() for kw in keywords)
        )
        df_domain = df_technical[mask].copy()
        
        if len(df_domain) > 0:
            domain_en = len(df_domain[df_domain['tech_preference'] == 'ENGLISH'])
            domain_zh = len(df_domain[df_domain['tech_preference'] == 'MANDARIN'])
            domain_total = domain_en + domain_zh
            
            if domain_total > 0:
                domain_en_pct = domain_en / domain_total
                domain_results[domain_name] = {
                    'english': domain_en,
                    'mandarin': domain_zh,
                    'total': domain_total,
                    'english_pct': domain_en_pct
                }
    
    # Print domain results sorted by English percentage (highest first)
    sorted_domains = sorted(
        domain_results.items(),
        key=lambda x: x[1]['english_pct'],
        reverse=True
    )
    
    if sorted_domains:
        print(f"{'Domain':<20} {'English':>8} {'Mandarin':>8} {'Total':>8} {'EN %':>8}")
        for domain, stats in sorted_domains:
            print(
                f"{domain:<20} {stats['english']:>8} {stats['mandarin']:>8} "
                f"{stats['total']:>8} {stats['english_pct']*100:>7.1f}%"
            )
    else:
        print("No domain-specific data found.")
    
    print("SAMPLE CODE-SWITCHED TECHNICAL TURNS")
    
    print("ENGLISH PREFERENCE (top 5):")
    english_samples = df_technical[df_technical['tech_preference'] == 'ENGLISH'].head(5)
    for idx, row in english_samples.iterrows():
        print(f"  [{row['speaker']}] {row['utterance'][:70]}...")
    
    print("MANDARIN PREFERENCE (top 5):")
    mandarin_samples = df_technical[df_technical['tech_preference'] == 'MANDARIN'].head(5)
    for idx, row in mandarin_samples.iterrows():
        print(f"  [{row['speaker']}] {row['utterance'][:70]}...")
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Plot 1: Overall distribution
    ax1 = axes[0, 0]
    labels = ['English\nPreferred', 'Mandarin\nPreferred']
    sizes = [english_count, mandarin_count]
    colors = ['#4C72B0', '#55A868']
    ax1.pie(sizes, labels=labels, autopct='%1.1f%%', colors=colors, startangle=90)
    ax1.set_title('Technical Term Preference\n(Code-Switched Technical Turns)', 
                  fontweight='bold', fontsize=12)
    
    # Plot 2: Authority skew
    ax2 = axes[0, 1]
    categories = ['English\nPreferred', 'Mandarin\nPreferred']
    percentages = [tjai*100, (1-tjai)*100]
    bars = ax2.bar(categories, percentages, color=['#4C72B0', '#55A868'], alpha=0.8)
    ax2.axhline(y=50, color='red', linestyle='--', linewidth=2, label='Balanced (50%)')
    ax2.set_ylabel('Percentage (%)', fontweight='bold')
    ax2.set_title('Authority Skew: English vs Mandarin', fontweight='bold', fontsize=12)
    ax2.set_ylim(0, 100)
    ax2.legend()
    for bar in bars:
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.1f}%', ha='center', va='bottom', fontweight='bold')
    
    # Plot 3: Domain-specific
    ax3 = axes[1, 0]
    if sorted_domains:
        domain_names = [d[0] for d in sorted_domains]
        domain_en_pcts = [d[1]['english_pct']*100 for d in sorted_domains]
        bars = ax3.barh(domain_names, domain_en_pcts, color='#4C72B0', alpha=0.8)
        ax3.axvline(x=50, color='red', linestyle='--', linewidth=2, label='Balanced')
        ax3.set_xlabel('English Preference (%)', fontweight='bold')
        ax3.set_title('English Authority by Technical Domain', fontweight='bold', fontsize=12)
        ax3.set_xlim(0, 100)
        ax3.legend()
        for i, bar in enumerate(bars):
            width = bar.get_width()
            ax3.text(width + 1, bar.get_y() + bar.get_height()/2.,
                    f'{width:.1f}%', ha='left', va='center', fontsize=9)
    else:
        ax3.text(0.5, 0.5, 'No domain data', ha='center', va='center')
    
    # Plot 4: Summary
    ax4 = axes[1, 1]
    ax4.axis('off')
    
    summary_text = f"""
SUMMARY STATISTICS

Total Code-Switched Turns: {len(df_code_switched):,}
Technical Content Turns: {len(df_technical):,}

English Preference: {english_count:,} ({tjai*100:.1f}%)
Mandarin Preference: {mandarin_count:,} ({(1-tjai)*100:.1f}%)

Authority Ratio: {authority_ratio:.2f}x

Statistical Tests:
  χ² = {chi2_stat:.2f}
  p-value = {p_value:.2e}
  Cramér's V = {cramers_v:.3f}

Interpretation:
  {'✓ SIGNIFICANT' if p_value < 0.05 else '✗ NOT SIGNIFICANT'}
  {'Weak' if cramers_v < 0.1 else 'Small' if cramers_v < 0.3 else 'Medium' if cramers_v < 0.5 else 'Large'} effect size
"""
    
    ax4.text(0.05, 0.95, summary_text, transform=ax4.transAxes,
            fontsize=11, verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    plt.savefig('mandarin_english_authority_skew.png', dpi=300, bbox_inches='tight')
    print("Visualization saved as 'mandarin_english_authority_skew.png'")
    plt.show()
    
    # Return results
    return {
        'total_code_switched': len(df_code_switched),
        'total_technical': len(df_technical),
        'english_count': english_count,
        'mandarin_count': mandarin_count,
        'tjai': tjai,
        'authority_ratio': authority_ratio,
        'chi2': chi2_stat,
        'p_value': p_value,
        'cramers_v': cramers_v,
        'domains': domain_results,
        'df_technical': df_technical
    }

if __name__ == '__main__':
    results = analyze_authority_skew(DATA_PATH)
    print("ANALYSIS COMPLETE")