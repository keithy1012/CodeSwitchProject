import os
import json
import re
import math
from collections import Counter, defaultdict
from datetime import date

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import ast

from sentence_transformers import SentenceTransformer, util
import torch

DATA_PATH = 'data/processed/2025-11-20/processed_dataset.csv'


# Define what we mean by "technical", "cultural", and "affective" using natural language
CONCEPT_ANCHORS = {
    'technical': [
        'This discusses software architecture, engineering, or development',
        'This discusses deployment, databases, APIs, or code',
        'This discusses performance, optimization, or technical infrastructure',
        '这讨论软件架构、工程或开发',
        '这讨论部署、数据库、API或代码',
        '这讨论性能、优化或技术基础设施'
    ],
    'cultural': [
        'This discusses traditions, customs, or heritage',
        'This discusses festivals, ceremonies, or celebrations',
        'This discusses family values, cultural identity, or cultural practices',
        '这讨论传统、习俗或文化遗产',
        '这讨论节日、仪式或庆祝活动',
        '这讨论家庭价值观、文化认同或文化实践'
    ],
    'affective': [
        'This expresses emotion, feeling, or sentiment',
        'This shows happiness, sadness, frustration, or emotional state',
        'This contains words about love, hate, joy, or other emotions',
        '这表达情感、感受或观点',
        '这表现出幸福、悲伤、沮丧或情感状态',
        '这包含关于爱、恨、喜悦或其他情感的词汇'
    ]
}

# Technical vocabulary (for fallback/validation)
TECH_WORDS = {
    'bug', 'deploy', 'api', 'architecture', 'database', 'model', 'epoch', 'latency',
    'optimize', 'merge', 'pull', 'request', 'code', 'repo', 'pr', 'feature', 'staging',
    'qa', 'deadline', 'build', 'server', 'client', 'framework', 'library', 'function',
    'class', 'variable', 'debug', 'test', 'branch', 'commit', 'refactor', 'performance',
    'cache', 'queue', 'async', 'thread', 'pipeline', 'docker', 'kubernetes', 'cloud',
    'aws', 'azure', 'gcp', 'ci', 'cd', 'devops', 'monitoring', 'logging', 'metric',
    'alert', 'deployment', 'production', 'algorithm', 'complexity', 'cpu', 'gpu', 'cuda',
    'interface', 'module', 'package', 'dependency', 'release', 'version', 'patch', 'upgrade',
    'downtime', 'uptime', 'redundancy', 'failover', 'backup', 'recovery', 'scalability',
    'throughput', 'bandwidth', 'protocol', 'encryption', 'authentication', 'authorization'
}

TECHNICAL_KEYWORDS = {
    'bug', 'api', 'model', 'deploy', 'code', 'pr', 'repo', 'build', 'database', 'server',
    'client', 'framework', 'library', 'function', 'class', 'variable', 'debug', 'test',
    'merge', 'branch', 'commit', 'pull', 'request', 'refactor', 'optimize', 'performance',
    'latency', 'cache', 'queue', 'async', 'thread', 'architecture', 'pipeline', 'docker',
    'kubernetes', 'cloud', 'aws', 'azure', 'gcp', 'ci', 'cd', 'devops', 'monitoring',
    'logging', 'metric', 'alert', 'deployment', 'staging', 'production', 'algorithm',
    'complexity', 'cpu', 'gpu', 'cuda', 'interface', 'module', 'package', 'dependency',
    'release', 'version', 'patch', 'upgrade', 'downtime', 'uptime', 'redundancy'
}

AFFECTIVE_KEYWORDS = {
    'love', 'happy', 'angry', 'sad', 'hate', 'joy', 'fear', 'disgust', 'surprise',
    'awesome', 'terrible', 'wonderful', 'horrible', 'amazing', 'disappointed', 'excited',
    'frustrated', 'thrilled', 'devastated', 'anxious', 'proud', 'embarrassed', 'grateful',
    'resentful', 'delighted', 'miserable', 'furious', 'cheerful', 'gloomy', 'ecstatic',
    'depressed', 'relieved', 'worried', 'confident', 'insecure', 'passionate', 'indifferent',
    'touched', 'hurt', 'blessed', 'cursed', 'inspired', 'demotivated', 'energized',
    'exhausted', 'hopeful', 'hopeless', 'optimistic', 'pessimistic', 'content', 'dissatisfied'
}

# ------------------
# SEMANTIC MODEL INITIALIZATION

class SemanticAnalyzer:
    """
    Uses sentence-transformers for semantic similarity analysis.
    This replaces hardcoded word lists with semantic understanding.
    """
    
    def __init__(self, model_name='embaas/sentence-transformers-multilingual-e5-base'):
        """Initialize the semantic model (downloads on first run, cached after)"""
        print(f"Loading semantic model: {model_name}")
        self.model = SentenceTransformer(model_name)
        print("Model loaded. Encoding concept anchors...")
        
        # Encode all concept anchors once (expensive operation, done once)
        self.concept_embeddings = {}
        for concept, texts in CONCEPT_ANCHORS.items():
            embeddings = self.model.encode(texts, convert_to_tensor=True)
            self.concept_embeddings[concept] = embeddings
        
        print("Concept anchors encoded. Ready for analysis.")
    
    def score_text(self, text, threshold=0.5):
        """
        Score a text for each concept (technical, cultural, affective).
        
        Returns:
            dict with scores for each concept
            {'technical': 0.75, 'cultural': 0.22, 'affective': 0.68}
        """
        if not text or len(str(text).strip()) == 0:
            return {'technical': 0.0, 'cultural': 0.0, 'affective': 0.0}
        
        text_embedding = self.model.encode(str(text), convert_to_tensor=True)
        scores = {}
        
        for concept, anchor_embeddings in self.concept_embeddings.items():
            # Compute similarity to all anchors for this concept
            similarities = util.pytorch_cos_sim(text_embedding, anchor_embeddings)
            # Take the max similarity (text is similar to at least one anchor)
            max_similarity = similarities.max().item()
            scores[concept] = round(max_similarity, 3)
        
        return scores

# Global analyzer instance
semantic_analyzer = None

def init_semantic_analyzer():
    """Initialize semantic analyzer on first run"""
    global semantic_analyzer
    if semantic_analyzer is None:
        semantic_analyzer = SemanticAnalyzer()
    return semantic_analyzer

def load_processed_dataset(path):
    """Load a processed CSV where each row is a conversation turn."""
    if not os.path.exists(path):
        print('Data file not found:', path)
        return []
    df = pd.read_csv(path, encoding='utf-8')

    def _parse_list_cell(x):
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

    if 'tokens_list' in df.columns:
        df['tokens_list'] = df['tokens_list'].apply(_parse_list_cell)
    if 'token_langs' in df.columns:
        df['token_langs'] = df['token_langs'].apply(_parse_list_cell)
    if 'token_count' in df.columns:
        df['token_count'] = pd.to_numeric(df['token_count'], errors='coerce').fillna(0).astype(int)

    if 'utterance' in df.columns and 'text' not in df.columns:
        df['text'] = df['utterance']
    if 'turn_index' in df.columns and 'turn' not in df.columns:
        df['turn'] = df['turn_index']
    if 'id' in df.columns and 'conv_id' not in df.columns:
        df['conv_id'] = df['id']
    if 'tokens' in df.columns and 'tokens_list' not in df.columns:
        df['tokens_list'] = df['tokens'].apply(_parse_list_cell)
    if 'Language' in df.columns and 'token_langs' not in df.columns:
        df['token_langs'] = df['Language'].apply(_parse_list_cell)
    if 'labels' in df.columns and 'lang' not in df.columns:
        def _labels_to_lang(x):
            try:
                v = ast.literal_eval(str(x))
                if isinstance(v, (set, list, tuple)):
                    s = set(v)
                elif isinstance(v, str) and v.startswith('{'):
                    return None
                else:
                    return None
            except Exception:
                s = set()
                sx = str(x)
                if 'en' in sx:
                    s.add('en')
                if 'zh' in sx or 'cn' in sx:
                    s.add('zh')
            if 'en' in s and 'zh' in s:
                return 'mix'
            if 'zh' in s:
                return 'zh'
            if 'en' in s:
                return 'en'
            return 'und'
        df['lang'] = df['labels'].apply(_labels_to_lang)

    if 'conv_id' not in df.columns:
        df['conv_id'] = 0

    convs = []
    for conv_id, group in df.groupby('conv_id'):
        g = group.sort_values('turn') if 'turn' in group.columns else group
        convs.append(g.reset_index(drop=True))
    return convs

# ------------------
# Core analysis functions
def analyze_record(rec, rec_id=None, topic_label=None, semantic_analyzer=None):
    """
    Analyze a single conversation record with semantic scoring.
    """
    if isinstance(rec, pd.DataFrame):
        df = rec.copy()
        def _ensure_list(x):
            if isinstance(x, (list, tuple, set)):
                return list(x)
            try:
                if pd.isna(x):
                    return []
            except Exception:
                pass
            s = str(x)
            try:
                v = ast.literal_eval(s)
                if isinstance(v, list):
                    return v
            except Exception:
                pass
            if ',' in s:
                return [p.strip() for p in s.split(',') if p.strip()]
            return [s]

        if 'token_langs' in df.columns:
            df['token_langs'] = df['token_langs'].apply(_ensure_list)
        else:
            df['token_langs'] = [[] for _ in range(len(df))]

        token_counts = defaultdict(int)
        for token_langs in df['token_langs']:
            for t in token_langs:
                token_counts[t] += 1

        turn_counts = df['lang'].value_counts().to_dict() if 'lang' in df.columns else {}
    else:
        rows = []
        try:
            for i, t in enumerate(rec):
                if isinstance(t, dict):
                    speaker = t.get('speaker', 'S')
                    text = t.get('text', '')
                elif isinstance(t, (list, tuple)) and len(t) >= 2:
                    speaker, text = t[0], t[1]
                else:
                    speaker, text = 'S', str(t)
                rows.append({'conv_id': rec_id, 'turn': i, 'speaker': speaker, 'text': text, 'lang': None, 'tokens_list': [], 'token_langs': [], 'token_count': 0})
        except Exception:
            rows = []
        df = pd.DataFrame(rows)
        token_counts = defaultdict(int)
        turn_counts = {}

    if semantic_analyzer:
        print(f"Scoring conversation {rec_id} with semantic model...")
        concept_scores = df['text'].apply(lambda x: semantic_analyzer.score_text(x))
        df[['tech_score_semantic', 'cultural_score_semantic', 'affective_score_semantic']] = \
            pd.DataFrame(concept_scores.tolist(), index=df.index)
        
        # Also keep keyword-based scores for comparison
        def tech_score(text):
            toks = [w.lower() for w in re.findall(r"\w+", text)]
            return sum(1 for w in toks if w in TECH_WORDS)
        
        df['tech_score_keyword'] = df['text'].apply(tech_score)
    else:
        # Fallback to keyword-based if semantic model not available
        def tech_score(text):
            toks = [w.lower() for w in re.findall(r"\w+", text)]
            return sum(1 for w in toks if w in TECH_WORDS)
        
        df['tech_score_semantic'] = 0.0
        df['cultural_score_semantic'] = 0.0
        df['affective_score_semantic'] = 0.0
        df['tech_score_keyword'] = df['text'].apply(tech_score)

    # Group technical scores by speaker (using semantic scores)
    tech_by_speaker = {}
    for speaker, group in df.groupby('speaker'):
        tech_total = group['tech_score_semantic'].sum()
        tech_by_speaker[speaker] = {
            'total_semantic': round(tech_total, 2),
            'avg_semantic_per_turn': round(tech_total / max(1, len(group)), 2)
        }

    df['is_mixed_turn'] = df['token_langs'].apply(lambda langs: (isinstance(langs, (list, tuple)) and ('zh' in langs) and ('en' in langs)))

    topic = topic_label or (rec.get('topic') if isinstance(rec, dict) else None)

    return {
        'conv_id': rec_id,
        'tokens': dict(token_counts),
        'turns': turn_counts,
        'tech_by_speaker': dict(tech_by_speaker),
        'topic': topic,
        'df': df
    }

def analyze_dataset(records, semantic_analyzer=None):
    analyses = []
    for i, rec in enumerate(records):
        rid = i
        if isinstance(rec, pd.DataFrame) and 'conv_id' in rec.columns:
            try:
                rid = int(rec['conv_id'].iloc[0])
            except Exception:
                rid = i
        analyses.append(analyze_record(rec, rec_id=rid, semantic_analyzer=semantic_analyzer))
    return analyses

# ------------------
# Bias Detection Functions (with Semantic Scoring)

def role_expertise_bias(analyses):
    """
    Role/Expertise Bias Analysis using semantic scores.
    """
    results = []
    
    for a in analyses:
        df = a['df'].copy()
        
        tech_by_lang = {}
        for lang, group in df.groupby('lang'):
            if not group.empty:
                # Use semantic scores
                tech_total = group['tech_score_semantic'].sum()
                tech_avg = group['tech_score_semantic'].mean()
                turn_count = len(group)
                tech_by_lang[lang] = {
                    'total_semantic': round(tech_total, 2),
                    'avg_per_turn_semantic': round(tech_avg, 3),
                    'turns': turn_count,
                    'tech_density_semantic': round(tech_total / max(1, turn_count), 3)
                }
        
        lang_densities = [v['tech_density_semantic'] for v in tech_by_lang.values() if v['tech_density_semantic'] > 0]
        expertise_imbalance = round(max(lang_densities) / min(lang_densities), 2) if len(lang_densities) > 1 else 1.0
        
        dominant_tech_lang = max(tech_by_lang, key=lambda k: tech_by_lang[k]['tech_density_semantic']) if tech_by_lang else None
        
        disadvantaged_langs = [lang for lang, metrics in tech_by_lang.items() 
                              if metrics['tech_density_semantic'] == 0 and lang != 'und']
        
        results.append({
            'conv_id': a['conv_id'],
            'tech_by_lang': tech_by_lang,
            'expertise_imbalance_ratio': expertise_imbalance,
            'dominant_tech_lang': dominant_tech_lang,
            'disadvantaged_langs': disadvantaged_langs,
            'bias_direction': f"{dominant_tech_lang} favored over {', '.join(disadvantaged_langs)}" if disadvantaged_langs else "balanced"
        })
    
    return results


def location_cultural_bias(analyses):
    """
    Location/Cultural Bias Analysis using semantic scores.
    """
    results = []
    
    for a in analyses:
        df = a['df'].copy()
        
        cult_by_lang = {}
        sent_by_lang = {}
        auth_by_lang = {}
        
        for lang, group in df.groupby('lang'):
            if not group.empty:
                # Use semantic cultural scores
                cult_total = group['cultural_score_semantic'].sum()
                cult_density = round(cult_total / max(1, len(group)), 3)
                cult_by_lang[lang] = {
                    'total_semantic': round(cult_total, 2),
                    'density_per_turn_semantic': cult_density,
                    'turns': len(group)
                }
                
                # Sentiment from affective scores
                affective_total = group['affective_score_semantic'].sum()
                affective_density = round(affective_total / max(1, len(group)), 3)
                sent_by_lang[lang] = {
                    'affective_density': affective_density,
                    'total_affective': round(affective_total, 2)
                }
                
                # Authenticity: cultural relative to technical
                tech_total = group['tech_score_semantic'].sum()
                if tech_total + cult_total > 0:
                    auth_by_lang[lang] = round(cult_total / (tech_total + cult_total), 3)
                else:
                    auth_by_lang[lang] = 0.0
        
        tokenism_flags = []
        for lang, metrics in cult_by_lang.items():
            if metrics['density_per_turn_semantic'] > 0.5:  # Semantic threshold
                tokenism_flags.append(f"{lang}: high cultural density ({metrics['density_per_turn_semantic']})")
        
        results.append({
            'conv_id': a['conv_id'],
            'cultural_density_by_lang': cult_by_lang,
            'affective_density_by_lang': sent_by_lang,
            'cultural_authenticity_ratio': auth_by_lang,
            'potential_tokenism': tokenism_flags if tokenism_flags else [],
            'cultural_suppression': {lang: metrics['density_per_turn_semantic'] == 0 
                                     for lang, metrics in cult_by_lang.items()}
        })
    
    return results

def topic_domain_bias(analyses):
    """
    Topic/Domain Bias Analysis using semantic scores.
    """
    results = []
    
    for a in analyses:
        df = a['df'].copy()
        
        topic_by_lang = {}
        
        for lang, group in df.groupby('lang'):
            if group.empty:
                continue
            
            tech_scores = group['tech_score_semantic'].tolist()
            cult_scores = group['cultural_score_semantic'].tolist()
            
            tech_total = sum(tech_scores)
            cult_total = sum(cult_scores)
            total_domain_refs = tech_total + cult_total
            
            if total_domain_refs > 0:
                tech_ratio = round(tech_total / total_domain_refs, 3)
                cult_ratio = round(cult_total / total_domain_refs, 3)
            else:
                tech_ratio = 0.0
                cult_ratio = 0.0
            
            domain_rigidity = round(max(tech_ratio, cult_ratio), 3)
            
            # Topic entropy
            if tech_ratio > 0 and cult_ratio > 0:
                topic_entropy = -(tech_ratio * math.log(tech_ratio) + cult_ratio * math.log(cult_ratio)) / math.log(2)
            else:
                topic_entropy = 0.0
            topic_entropy = round(topic_entropy, 3)
            
            mixed_turns = sum(1 for t, c in zip(tech_scores, cult_scores) if t > 0.3 and c > 0.3)
            turns_with_domain = sum(1 for t, c in zip(tech_scores, cult_scores) if t > 0.3 or c > 0.3)
            
            if turns_with_domain > 0:
                topic_diversity = round(mixed_turns / turns_with_domain, 3)
            else:
                topic_diversity = 0.0
            
            topic_by_lang[lang] = {
                'tech_ratio': tech_ratio,
                'cultural_ratio': cult_ratio,
                'domain_rigidity': domain_rigidity,
                'topic_diversity': topic_diversity,
                'topic_entropy': topic_entropy,
                'turns_analyzed': len(group),
                'dominant_domain': 'technical' if tech_ratio > cult_ratio else 'cultural' if cult_ratio > tech_ratio else 'balanced'
            }
        
        rigidities = [v['domain_rigidity'] for v in topic_by_lang.values()]
        avg_rigidity = round(sum(rigidities) / max(1, len(rigidities)), 3) if rigidities else 0.0
        
        if avg_rigidity > 0.8:
            confinement = 'SEVERE'
        elif avg_rigidity > 0.7:
            confinement = 'HIGH'
        elif avg_rigidity > 0.55:
            confinement = 'MEDIUM'
        else:
            confinement = 'LOW'
        
        results.append({
            'conv_id': a['conv_id'],
            'topic_focus_by_lang': topic_by_lang,
            'avg_domain_rigidity': avg_rigidity,
            'potential_confinement': confinement
        })
    
    return results

def compute_switches_for_conversation(df):
    """
    Improved functional switch counting using semantic scores.
    """
    intra = 0
    inter = 0
    functional_counts = Counter()
    mixed_function_turns = 0

    for _, row in df.iterrows():
        langs = set()
        if 'token_langs' in row and isinstance(row['token_langs'], (list, tuple)):
            langs = set(row['token_langs'])
        else:
            txt = str(row.get('text', ''))
            if re.search(r'[\u4e00-\u9fff]', txt):
                langs.add('zh')
            if re.search(r'[A-Za-z]', txt):
                langs.add('en')

        if 'zh' in langs and 'en' in langs:
            intra += 1
            
            # Use semantic scores instead of keyword matching
            tech_score = row.get('tech_score_semantic', 0)
            affective_score = row.get('affective_score_semantic', 0)
            
            has_technical = tech_score > 0.4
            has_affective = affective_score > 0.4
            
            if has_technical:
                functional_counts['technical'] += 1
            if has_affective:
                functional_counts['affective'] += 1
            if has_technical and has_affective:
                mixed_function_turns += 1
            if not has_technical and not has_affective:
                functional_counts['other'] += 1

    langs = list(df['lang'])
    for a, b in zip(langs, langs[1:]):
        if a != b and a != 'und' and b != 'und':
            inter += 1

    return {
        'intra': intra,
        'inter': inter,
        'functional_counts': dict(functional_counts),
        'mixed_function_turns': mixed_function_turns
    }

def functional_switch_skew(analyses):
    agg = Counter()
    total_mixed = 0
    for a in analyses:
        stats = compute_switches_for_conversation(a['df'])
        agg.update(stats.get('functional_counts', {}))
        total_mixed += stats.get('mixed_function_turns', 0)
    return dict(agg), total_mixed

# ------------------
# Visualization helpers
def plot_language_tokens(analyses, show=True):
    tot = defaultdict(int)
    for a in analyses:
        for k, v in a['tokens'].items():
            tot[k] += v
    if not tot:
        print('No token data available to plot.')
        return
    items = sorted(tot.items(), key=lambda x: -x[1])
    langs = [x[0] for x in items]
    vals = [x[1] for x in items]
    plt.figure(figsize=(6,3))
    plt.bar(langs, vals, color=['#4C72B0', '#55A868', '#C44E52', '#8c8c8c'])
    plt.title('Token counts per detected language')
    plt.tight_layout()
    if show:
        plt.show()

def plot_switch_counts(analyses, show=True):
    ins = []
    inters = []
    for a in analyses:
        stats = compute_switches_for_conversation(a['df'])
        ins.append(stats['intra'])
        inters.append(stats['inter'])
    if len(ins) == 0 and len(inters) == 0:
        print('No switch data to plot.')
        return
    df_plot = pd.DataFrame({'intra': ins, 'inter': inters})
    df_plot.index.name = 'conv'
    df_plot.plot(kind='bar', stacked=False, figsize=(8,3), color=['#C44E52', '#4C72B0'])
    plt.title('Per-conversation intra vs inter switch counts')
    plt.tight_layout()
    if show:
        plt.show()

# ------------------
# Export helpers
def export_summary(analyses, out_dir=f'data/processed/{date.today().strftime("%Y-%m-%d")}/assignment5_output'):
    os.makedirs(out_dir, exist_ok=True)
    for a in analyses:
        df = a['df'].copy()
        conv_id = a.get('conv_id', 'unknown')
        df.to_csv(os.path.join(out_dir, f'conv_{conv_id}_turns.csv'), index=False, encoding='utf-8')
    
    func_skew, mixed_func = functional_switch_skew(analyses)
    summary = {
        'num_conversations': len(analyses),
        'functional_switch_skew': func_skew,
        'mixed_function_turns': mixed_func,
        'date': date.today().isoformat(),
        'note': 'Semantic analysis using sentence-transformers/multilingual-e5-large'
    }
    with open(os.path.join(out_dir, 'summary.json'), 'w', encoding='utf-8') as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print('Exported analysis to', out_dir)

# ------------------
# Improved Reporting
def print_bias_summary(expertise_results, cultural_results, domain_results, use_semantic=True):
    """Print aggregate summary with semantic or keyword-based findings."""
    print('\n' + '='*70)
    if use_semantic:
        print('AGGREGATE BIAS ANALYSIS SUMMARY (SEMANTIC - using multilingual-e5)')
    else:
        print('AGGREGATE BIAS ANALYSIS SUMMARY (KEYWORD-BASED)')
    print('='*70 + '\n')
    
    num_convs = len(expertise_results)
    
    # Role/Expertise Bias
    print('1. ROLE/EXPERTISE BIAS SUMMARY')
    
    imbalance_ratios = [r['expertise_imbalance_ratio'] for r in expertise_results]
    avg_imbalance = sum(imbalance_ratios) / len(imbalance_ratios) if imbalance_ratios else 0
    
    all_tech_by_lang = defaultdict(list)
    for result in expertise_results:
        for lang, metrics in result['tech_by_lang'].items():
            key = 'tech_density_semantic' if use_semantic else 'tech_density'
            all_tech_by_lang[lang].append(metrics.get(key, metrics.get('tech_density_semantic', 0)))
    
    print(f"Total Conversations Analyzed: {num_convs}")
    print(f"Average Expertise Imbalance Ratio: {avg_imbalance:.2f}")
    print(f"  (Ratio > 1.5 indicates significant bias toward one language)")
    print(f"\nTechnical Density by Language (aggregate - SEMANTIC SCORES):")
    for lang in sorted(all_tech_by_lang.keys()):
        densities = all_tech_by_lang[lang]
        avg_density = sum(densities) / len(densities)
        print(f"  {lang}: avg={avg_density:.3f}, min={min(densities):.3f}, max={max(densities):.3f}")
    
    disadvantaged_count = defaultdict(int)
    for result in expertise_results:
        for lang in result['disadvantaged_langs']:
            disadvantaged_count[lang] += 1
    
    if disadvantaged_count:
        print(f"\nLanguages Disadvantaged in Technical Discourse:")
        for lang, count in sorted(disadvantaged_count.items(), key=lambda x: -x[1]):
            print(f"  {lang}: excluded from tech in {count}/{num_convs} conversations ({count/num_convs*100:.1f}%)")
    
    dominant_langs = [r['dominant_tech_lang'] for r in expertise_results if r['dominant_tech_lang']]
    if dominant_langs:
        dominant_count = Counter(dominant_langs)
        print(f"\nDominant Technical Language Distribution:")
        for lang, count in dominant_count.most_common():
            print(f"  {lang}: {count} conversations ({count/num_convs*100:.1f}%)")
    
    # Location/Cultural Bias
    print('\n\n2. LOCATION/CULTURAL BIAS SUMMARY (SEMANTIC)')
    
    all_cult_by_lang = defaultdict(list)
    all_affective_by_lang = defaultdict(list)
    all_auth_by_lang = defaultdict(list)
    tokenism_count = 0
    cultural_suppression_count = defaultdict(int)
    
    for result in cultural_results:
        for lang, metrics in result['cultural_density_by_lang'].items():
            all_cult_by_lang[lang].append(metrics.get('density_per_turn_semantic', 0))
        for lang, metrics in result['affective_density_by_lang'].items():
            all_affective_by_lang[lang].append(metrics.get('affective_density', 0))
        for lang, ratio in result['cultural_authenticity_ratio'].items():
            all_auth_by_lang[lang].append(ratio)
        tokenism_count += len(result['potential_tokenism'])
        for lang, suppressed in result['cultural_suppression'].items():
            if suppressed:
                cultural_suppression_count[lang] += 1
    
    print(f"Total Conversations Analyzed: {num_convs}")
    print(f"Conversations with High Cultural Density (semantic): {tokenism_count}")
    
    if cultural_suppression_count:
        print(f"\nLanguages with Suppressed Cultural Content (SEMANTIC):")
        for lang, count in sorted(cultural_suppression_count.items(), key=lambda x: -x[1]):
            print(f"  {lang}: zero cultural refs in {count}/{num_convs} conversations ({count/num_convs*100:.1f}%)")
    
    print(f"\nCultural Density by Language (SEMANTIC - refs per turn):")
    for lang in sorted(all_cult_by_lang.keys()):
        densities = all_cult_by_lang[lang]
        if densities:
            avg_density = sum(densities) / len(densities)
            print(f"  {lang}: avg={avg_density:.3f}, min={min(densities):.3f}, max={max(densities):.3f}")
    
    print(f"\nAffective Density by Language (SEMANTIC - emotion per turn):")
    for lang in sorted(all_affective_by_lang.keys()):
        affectives = all_affective_by_lang[lang]
        if affectives:
            avg_aff = sum(affectives) / len(affectives)
            print(f"  {lang}: avg={avg_aff:.3f}, min={min(affectives):.3f}, max={max(affectives):.3f}")
    
    print(f"\nCultural Authenticity Ratio (cultural/(cultural+technical) - SEMANTIC):")
    for lang in sorted(all_auth_by_lang.keys()):
        ratios = all_auth_by_lang[lang]
        if ratios:
            avg_ratio = sum(ratios) / len(ratios)
            print(f"  {lang}: avg={avg_ratio:.3f}")
            print(f"    → {avg_ratio:.1%} of domain references are cultural (vs technical)")
    
    # Topic/Domain Bias
    print('\n\n3. TOPIC/DOMAIN BIAS SUMMARY (SEMANTIC)')
    
    all_rigidities = []
    all_entropies = defaultdict(list)
    all_diversities = defaultdict(list)
    confinement_levels = Counter()
    
    for result in domain_results:
        all_rigidities.append(result['avg_domain_rigidity'])
        confinement_levels[result['potential_confinement']] += 1
        for lang, metrics in result['topic_focus_by_lang'].items():
            all_diversities[lang].append(metrics['topic_diversity'])
            all_entropies[lang].append(metrics['topic_entropy'])
    
    avg_rigidity = sum(all_rigidities) / len(all_rigidities) if all_rigidities else 0
    
    print(f"Total Conversations Analyzed: {num_convs}")
    print(f"Average Domain Rigidity (aggregate - SEMANTIC): {avg_rigidity:.3f}")
    print(f"  (0.5 = balanced, 1.0 = completely rigid)")
    
    print(f"\nConfinement Level Distribution:")
    for level in ['SEVERE', 'HIGH', 'MEDIUM', 'LOW']:
        count = confinement_levels[level]
        pct = count / num_convs * 100 if num_convs > 0 else 0
        if count > 0:
            print(f"  {level}: {count} conversations ({pct:.1f}%)")
    
    print(f"\nTopic Entropy by Language (SEMANTIC - higher = more balanced):")
    for lang in sorted(all_entropies.keys()):
        entropies = all_entropies[lang]
        if entropies:
            avg_ent = sum(entropies) / len(entropies)
            print(f"  {lang}: avg={avg_ent:.3f}, min={min(entropies):.3f}, max={max(entropies):.3f}")
    
    print(f"\nTopic Diversity by Language (SEMANTIC - % mixed turns):")
    for lang in sorted(all_diversities.keys()):
        diversities = all_diversities[lang]
        if diversities:
            avg_div = sum(diversities) / len(diversities)
            print(f"  {lang}: avg={avg_div:.3f}")
            print(f"    → {avg_div:.1%} of turns mix technical and cultural topics")
    
    # Overall Risk Assessment
    print('\n\n4. OVERALL ETHICAL AI BIAS RISK ASSESSMENT (SEMANTIC)')
    
    risk_score = 0
    risk_factors = []
    
    if avg_imbalance > 1.5:
        risk_score += 3
        risk_factors.append(f"HIGH expertise imbalance (ratio={avg_imbalance:.2f}) [+3]")
    elif avg_imbalance > 1.2:
        risk_score += 2
        risk_factors.append(f"MODERATE expertise imbalance (ratio={avg_imbalance:.2f}) [+2]")
    elif avg_imbalance > 1.0:
        risk_score += 1
        risk_factors.append(f"MILD expertise imbalance (ratio={avg_imbalance:.2f}) [+1]")
    
    if sum(disadvantaged_count.values()) > num_convs * 0.2:
        risk_score += 3
        risk_factors.append(f"Languages excluded from technical discourse [+3]")
    
    if sum(cultural_suppression_count.values()) > num_convs * 0.3:
        risk_score += 2
        risk_factors.append(f"Cultural suppression in {sum(cultural_suppression_count.values())}/{num_convs} conversations [+2]")
    
    high_or_severe = confinement_levels.get('HIGH', 0) + confinement_levels.get('SEVERE', 0)
    if high_or_severe > num_convs * 0.4:
        risk_score += 2
        risk_factors.append(f"High domain confinement ({high_or_severe} conversations) [+2]")
    
    if not risk_factors:
        risk_factors.append("No significant bias detected")
        risk_assessment = "LOW RISK"
    elif risk_score <= 2:
        risk_assessment = "LOW-MODERATE RISK"
    elif risk_score <= 4:
        risk_assessment = "MODERATE RISK"
    elif risk_score <= 6:
        risk_assessment = "HIGH RISK"
    else:
        risk_assessment = "CRITICAL RISK"
    
    print(f"Risk Assessment: {risk_assessment}")
    print(f"Risk Score: {risk_score}/9 (max)")
    print(f"\nKey Findings:")
    for i, factor in enumerate(risk_factors, 1):
        print(f"  {i}. {factor}")
    
    print('\n' + '='*70 + '\n')

def main():
    print("Initializing semantic analyzer...")
    semantic_analyzer = init_semantic_analyzer()
    
    print("\nLoading dataset...")
    records = load_processed_dataset(DATA_PATH)
    if not records:
        print('No records found; exiting.')
        return
    
    print(f"Analyzing {len(records)} conversations with semantic model...")
    analyses = analyze_dataset(records, semantic_analyzer=semantic_analyzer)
    print(f'Analyzed {len(analyses)} conversations.')

    total_tokens = Counter()
    total_turns = Counter()
    for a in analyses:
        for k, v in a['tokens'].items():
            total_tokens[k] += v
        for k, v in a['turns'].items():
            total_turns[k] += v

    print('Token totals by detected token language:', dict(total_tokens))
    print('Utterance-level turn totals by lang:', dict(total_turns))
    
    func_skew, mixed_func = functional_switch_skew(analyses)
    print('Functional switch skew (aggregate):', func_skew)
    print(f'Turns with mixed technical+affective content: {mixed_func}')
    
    print('\n BIAS DETECTION RESULTS (SEMANTIC ANALYSIS) \n')
    
    expertise_results = role_expertise_bias(analyses)
    cultural_results = location_cultural_bias(analyses)
    domain_results = topic_domain_bias(analyses)
    
    print_bias_summary(expertise_results, cultural_results, domain_results, use_semantic=True)

    export_summary(analyses)

    try:
        plot_language_tokens(analyses, show=True)
        plot_switch_counts(analyses, show=True)
    except Exception as e:
        print('Plotting error:', e)

if __name__ == '__main__':
    main()