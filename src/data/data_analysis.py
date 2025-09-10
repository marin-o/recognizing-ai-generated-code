import os
import sys
import argparse
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from collections import defaultdict
import warnings

warnings.filterwarnings('ignore')

class CoDeTDataAnalyzer:
    def __init__(self, data_path: str = "../../data/"):
        self.data_path = data_path
        self.train = None
        self.val = None
        self.test = None
        self.df_combined = None
        plt.style.use('default')
        sns.set_palette('husl')
        plt.rcParams['figure.figsize'] = (12, 8)
    
    def _safe_save_figure(self, fig, save_path: str, chart_name: str):
        try:
            fig.write_html(f"{save_path}/{chart_name}.html")
            fig.write_image(f"{save_path}/{chart_name}.png")
        except:
            fig.write_html(f"{save_path}/{chart_name}.html")
    
    def load_data(self):
        from dataset import CoDeTM4
        self.train, self.val, self.test = CoDeTM4(self.data_path).get_dataset(
            ['train', 'val', 'test'], columns='all', dynamic_split_sizing=False)
        self._create_combined_dataframe()
    
    def _create_combined_dataframe(self):
        train_df = self.train.to_pandas()
        train_df['split'] = 'Train'
        val_df = self.val.to_pandas()
        val_df['split'] = 'Validation'
        test_df = self.test.to_pandas()
        test_df['split'] = 'Test'
        self.df_combined = pd.concat([train_df, val_df, test_df], ignore_index=True)
        self.df_combined['code_length'] = self.df_combined['cleaned_code'].str.len()
        self.df_combined['num_lines'] = self.df_combined['cleaned_code'].str.count('\n') + 1
        self.df_combined['num_words'] = self.df_combined['cleaned_code'].str.split().str.len()
    
    def chart1_split_proportions(self, save_path=None):
        split_counts = self.df_combined['split'].value_counts()
        fig = px.pie(values=split_counts.values, names=split_counts.index, title="Dataset Split Proportions",
                     color_discrete_sequence=['#FF6B6B', '#4ECDC4', '#45B7D1'])
        fig.update_traces(textposition='inside', textinfo='percent+label')
        fig.update_layout(title_font_size=16, title_x=0.5, showlegend=True, width=600, height=500)
        if save_path:
            self._safe_save_figure(fig, save_path, "chart1_split_proportions")
        fig.show()
        return fig
    
    def chart2_language_distribution(self, save_path=None):
        lang_split_counts = self.df_combined.groupby(['split', 'language']).size().unstack(fill_value=0)
        # Reorder the index to ensure proper ordering
        lang_split_counts = lang_split_counts.reindex(['Train', 'Validation', 'Test'])
        fig = px.bar(lang_split_counts, title="Programming Language Distribution Across Dataset Splits",
                     labels={'value': 'Number of Samples', 'index': 'Dataset Split'},
                     color_discrete_sequence=['#FF9999', '#66B2FF', '#99FF99'])
        fig.update_layout(title_font_size=16, title_x=0.5, xaxis_title="Dataset Split", yaxis_title="Number of Samples",
                          legend_title="Programming Language", width=800, height=500, bargap=0.3)
        if save_path:
            self._safe_save_figure(fig, save_path, "chart2_language_distribution")
        fig.show()
        return fig
    
    def chart3_target_distribution(self, save_path=None):
        target_split_counts = self.df_combined.groupby(['split', 'target_binary']).size().unstack(fill_value=0)
        fig = go.Figure()
        splits = ['Train', 'Validation', 'Test']  # Fixed order
        colors = ['#FF6B6B', '#4ECDC4', '#45B7D1']
        for i, split in enumerate(splits):
            fig.add_trace(go.Bar(name=split, x=['Negative (0)', 'Positive (1)'],
                                 y=[target_split_counts.loc[split, 0], target_split_counts.loc[split, 1]],
                                 marker_color=colors[i], opacity=0.8))
        fig.update_layout(title="Target Class Distribution Across Dataset Splits", title_font_size=16, title_x=0.5,
                          xaxis_title="Target Class", yaxis_title="Number of Samples", barmode='group',
                          width=700, height=500, legend_title="Dataset Split")
        if save_path:
            self._safe_save_figure(fig, save_path, "chart3_target_distribution")
        fig.show()
        return fig
    
    def chart4_language_target_heatmap(self, save_path=None):
        lang_target_counts = self.df_combined.groupby(['language', 'target_binary']).size().unstack(fill_value=0)
        lang_target_pct = lang_target_counts.div(lang_target_counts.sum(axis=1), axis=0) * 100
        fig = px.imshow(lang_target_pct.values, x=['Negative (0)', 'Positive (1)'], y=lang_target_pct.index,
                        color_continuous_scale='RdYlBu_r', title="Target Class Distribution by Programming Language (%)",
                        text_auto='.1f')
        fig.update_layout(title_font_size=16, title_x=0.5, xaxis_title="Target Class", yaxis_title="Programming Language",
                          width=600, height=400)
        if save_path:
            self._safe_save_figure(fig, save_path, "chart4_language_target_heatmap")
        fig.show()
        return fig
    
    def chart5_code_length_distribution(self, save_path=None):
        fig = go.Figure()
        colors = ['#FF6B6B', '#4ECDC4', '#45B7D1']
        splits = ['Train', 'Validation', 'Test']
        for i, split in enumerate(splits):
            split_data = self.df_combined[self.df_combined['split'] == split]['code_length']
            fig.add_trace(go.Histogram(x=split_data, name=split, opacity=0.7, nbinsx=50, marker_color=colors[i]))
        fig.update_layout(title="Code Length Distribution Across Dataset Splits", title_font_size=16, title_x=0.5,
                          xaxis_title="Code Length (characters)", yaxis_title="Frequency", barmode='overlay',
                          width=900, height=500, legend_title="Dataset Split")
        if save_path:
            self._safe_save_figure(fig, save_path, "chart5_code_length_distribution")
        fig.show()
        return fig
    
    def chart6_model_distribution(self, save_path=None):
        if 'model' in self.df_combined.columns:
            model_counts = self.df_combined['model'].value_counts()
            fig = px.bar(x=model_counts.values, y=model_counts.index, orientation='h',
                         title="Model Distribution in Dataset", labels={'x': 'Number of Samples', 'y': 'Model'},
                         color=model_counts.values, color_continuous_scale='viridis')
            fig.update_layout(title_font_size=16, title_x=0.5, width=800, height=600, showlegend=False)
            if save_path:
                self._safe_save_figure(fig, save_path, "chart6_model_distribution")
            fig.show()
            return fig
        return None
    
    def chart7_model_split_distribution(self, save_path=None):
        if 'model' in self.df_combined.columns:
            model_split_counts = self.df_combined.groupby(['split', 'model']).size().unstack(fill_value=0)
            # Reorder the index to ensure proper ordering
            model_split_counts = model_split_counts.reindex(['Train', 'Validation', 'Test'])
            fig = px.bar(model_split_counts, title="Model Distribution Within Each Dataset Split",
                         labels={'value': 'Number of Samples', 'index': 'Dataset Split'},
                         color_discrete_sequence=px.colors.qualitative.Set3)
            fig.update_layout(title_font_size=16, title_x=0.5, xaxis_title="Dataset Split",
                              yaxis_title="Number of Samples", legend_title="Model",
                              width=1000, height=600, bargap=0.3)
            if save_path:
                self._safe_save_figure(fig, save_path, "chart7_model_split_distribution")
            fig.show()
            return fig
        return None
    
    def chart8_target_balance_analysis(self, save_path=None):
        target_percentages = self.df_combined.groupby('split')['target_binary'].agg(['mean', 'count']).reset_index()
        target_percentages['positive_pct'] = target_percentages['mean'] * 100
        target_percentages['negative_pct'] = (1 - target_percentages['mean']) * 100
        # Reorder to ensure proper order
        split_order = ['Train', 'Validation', 'Test']
        target_percentages['split'] = pd.Categorical(target_percentages['split'], categories=split_order, ordered=True)
        target_percentages = target_percentages.sort_values('split')
        fig = go.Figure()
        fig.add_trace(go.Bar(name='Negative (0)', x=target_percentages['split'],
                             y=target_percentages['negative_pct'], marker_color='#FF6B6B', opacity=0.8))
        fig.add_trace(go.Bar(name='Positive (1)', x=target_percentages['split'],
                             y=target_percentages['positive_pct'], marker_color='#4ECDC4', opacity=0.8))
        fig.update_layout(title="Target Class Balance Across Dataset Splits (%)", title_font_size=16, title_x=0.5,
                          xaxis_title="Dataset Split", yaxis_title="Percentage", barmode='stack',
                          width=700, height=500, legend_title="Target Class")
        for i, row in target_percentages.iterrows():
            fig.add_annotation(x=row['split'], y=row['negative_pct']/2, text=f"{row['negative_pct']:.1f}%",
                               showarrow=False, font=dict(color="white", size=12))
            fig.add_annotation(x=row['split'], y=row['negative_pct'] + row['positive_pct']/2,
                               text=f"{row['positive_pct']:.1f}%", showarrow=False, font=dict(color="white", size=12))
        if save_path:
            self._safe_save_figure(fig, save_path, "chart8_target_balance")
        fig.show()
        return fig
    
    def chart9_code_length_by_language(self, save_path=None):
        fig = px.box(self.df_combined, x='language', y='code_length',
                     title="Code Length Distribution by Programming Language", color='language',
                     color_discrete_sequence=['#FF6B6B', '#4ECDC4', '#45B7D1'])
        fig.update_layout(title_font_size=16, title_x=0.5, xaxis_title="Programming Language",
                          yaxis_title="Code Length (characters)", width=800, height=500, showlegend=False)
        if save_path:
            self._safe_save_figure(fig, save_path, "chart9_code_length_by_language")
        fig.show()
        return fig
    
    def chart10_language_split_sunburst(self, save_path=None):
        lang_split_counts = self.df_combined.groupby(['language', 'split']).size().reset_index(name='count')
        fig = px.sunburst(lang_split_counts, path=['language', 'split'], values='count',
                          title="Hierarchical View: Language → Split Distribution",
                          color_discrete_sequence=px.colors.qualitative.Set3)
        fig.update_layout(title_font_size=16, title_x=0.5, width=700, height=700)
        if save_path:
            self._safe_save_figure(fig, save_path, "chart10_language_split_sunburst")
        fig.show()
        return fig
    
    def chart11_summary_statistics_table(self, save_path=None):
        summary_stats = []
        for split in ['Train', 'Validation', 'Test']:
            split_data = self.df_combined[self.df_combined['split'] == split]
            stats = {
                'Split': split, 'Total Samples': len(split_data),
                'Python': len(split_data[split_data['language'] == 'python']),
                'Java': len(split_data[split_data['language'] == 'java']),
                'C++': len(split_data[split_data['language'] == 'cpp']),
                'Positive Class': len(split_data[split_data['target_binary'] == 1]),
                'Negative Class': len(split_data[split_data['target_binary'] == 0]),
                'Avg Code Length': split_data['code_length'].mean(),
                'Min Code Length': split_data['code_length'].min(),
                'Max Code Length': split_data['code_length'].max()
            }
            summary_stats.append(stats)
        summary_df = pd.DataFrame(summary_stats)
        fig = go.Figure(data=[go.Table(
            header=dict(values=list(summary_df.columns), fill_color='lightblue', align='center', font=dict(size=12)),
            cells=dict(values=[summary_df[col] for col in summary_df.columns], fill_color='white', align='center', font=dict(size=11)))])
        fig.update_layout(title="Dataset Statistics Summary", title_font_size=16, title_x=0.5, width=1200, height=400)
        if save_path:
            self._safe_save_figure(fig, save_path, "chart11_summary_table")
        fig.show()
        return fig, summary_df
    
    def chart12_correlation_matrix(self, save_path=None):
        numerical_cols = ['target_binary', 'code_length', 'num_lines', 'num_words']
        corr_matrix = self.df_combined[numerical_cols].corr()
        fig = px.imshow(corr_matrix, title="Correlation Matrix of Numerical Features",
                        color_continuous_scale='RdBu_r', aspect="auto", text_auto='.3f')
        fig.update_layout(title_font_size=16, title_x=0.5, width=600, height=500)
        if save_path:
            self._safe_save_figure(fig, save_path, "chart12_correlation_matrix")
        fig.show()
        return fig
    
    def chart13_data_leakage_analysis(self, save_path=None):
        duplicates_info = self._find_exact_duplicates()
        if not duplicates_info['exact_duplicates']:
            return None
        fig = make_subplots(rows=2, cols=2,
                            subplot_titles=('Overlap Types Distribution', 'Leakage Impact by Split',
                                            'Duplicate Code Length Distribution', 'Leakage Summary Statistics'),
                            specs=[[{"type": "bar"}, {"type": "bar"}], [{"type": "histogram"}, {"type": "table"}]])
        fig.add_trace(go.Bar(x=list(duplicates_info['overlap_counts'].keys()),
                             y=list(duplicates_info['overlap_counts'].values()),
                             name="Overlap Types", marker_color='#FF6B6B'), row=1, col=1)
        fig.add_trace(go.Bar(x=list(duplicates_info['leakage_impact'].keys()),
                             y=list(duplicates_info['leakage_impact'].values()),
                             name="Affected Samples", marker_color='#4ECDC4'), row=1, col=2)
        fig.add_trace(go.Histogram(x=duplicates_info['duplicate_lengths'], name="Code Length",
                                  marker_color='#45B7D1', nbinsx=30), row=2, col=1)
        fig.add_trace(go.Table(header=dict(values=['Metric', 'Value'], fill_color='lightblue', align='center'),
                              cells=dict(values=[list(duplicates_info['summary_stats'].keys()),
                                               list(duplicates_info['summary_stats'].values())],
                                        fill_color='white', align='center')), row=2, col=2)
        fig.update_layout(title="Data Leakage Analysis: Exact Matches Between Splits", title_font_size=16,
                          title_x=0.5, height=800, width=1200, showlegend=False)
        fig.update_xaxes(title_text="Overlap Type", row=1, col=1)
        fig.update_yaxes(title_text="Number of Duplicates", row=1, col=1)
        fig.update_xaxes(title_text="Dataset Split", row=1, col=2)
        fig.update_yaxes(title_text="Affected Samples", row=1, col=2)
        fig.update_xaxes(title_text="Code Length (characters)", row=2, col=1)
        fig.update_yaxes(title_text="Frequency", row=2, col=1)
        if save_path:
            self._safe_save_figure(fig, save_path, "chart13_data_leakage_analysis")
        fig.show()
        return fig, duplicates_info
    
    def _find_exact_duplicates(self):
        code_to_splits = defaultdict(list)
        for split_name in ['Train', 'Validation', 'Test']:
            split_data = self.df_combined[self.df_combined['split'] == split_name]
            for idx, row in split_data.iterrows():
                code_to_splits[row['cleaned_code']].append((split_name, idx))
        exact_duplicates = {}
        for code, occurrences in code_to_splits.items():
            if len(occurrences) > 1:
                splits_involved = set(split for split, _ in occurrences)
                if len(splits_involved) > 1:
                    exact_duplicates[code] = occurrences
        overlap_counts = defaultdict(int)
        leakage_impact = defaultdict(int)
        duplicate_lengths = []
        total_overlapping_samples = 0
        for code, occurrences in exact_duplicates.items():
            splits_involved = sorted(set(split for split, _ in occurrences))
            overlap_key = '-'.join(splits_involved)
            overlap_counts[overlap_key] += 1
            for split, _ in occurrences:
                leakage_impact[split] += 1
            duplicate_lengths.append(len(code))
            total_overlapping_samples += len(occurrences)
        summary_stats = {
            'Total Unique Duplicates': len(exact_duplicates),
            'Total Affected Samples': total_overlapping_samples,
            'Avg Duplicate Code Length': np.mean(duplicate_lengths) if duplicate_lengths else 0,
            'Max Duplicate Code Length': max(duplicate_lengths) if duplicate_lengths else 0,
            'Min Duplicate Code Length': min(duplicate_lengths) if duplicate_lengths else 0
        }
        return {
            'exact_duplicates': exact_duplicates,
            'overlap_counts': dict(overlap_counts),
            'leakage_impact': dict(leakage_impact),
            'duplicate_lengths': duplicate_lengths,
            'total_overlapping_samples': total_overlapping_samples,
            'summary_stats': summary_stats
        }
    
    def chart14_simple_data_leakage_breakdown(self, save_path=None):
        duplicates_info = self._find_exact_duplicates()
        exact_duplicates = duplicates_info['exact_duplicates']
        if not exact_duplicates:
            return None
        overlap_details = {'Train ↔ Validation': 0, 'Test ↔ Train': 0, 'Test ↔ Validation': 0, 'All Three Splits': 0}
        for code, occurrences in exact_duplicates.items():
            splits_involved = set(split for split, idx in occurrences)
            if len(splits_involved) == 3:
                overlap_details['All Three Splits'] += 1
            elif splits_involved == {'Train', 'Validation'}:
                overlap_details['Train ↔ Validation'] += 1
            elif splits_involved == {'Test', 'Train'}:
                overlap_details['Test ↔ Train'] += 1
            elif splits_involved == {'Test', 'Validation'}:
                overlap_details['Test ↔ Validation'] += 1
        fig = go.Figure(data=[go.Bar(x=list(overlap_details.keys()), y=list(overlap_details.values()),
                                     marker_color=['#FF6B6B', '#4ECDC4', '#45B7D1', '#FFA07A'],
                                     text=list(overlap_details.values()), textposition='inside', textfont_size=14, textangle=0)])
        fig.update_layout(title="Data Leakage: Detailed Split Overlap Analysis", title_font_size=16, title_x=0.5,
                          xaxis_title="Split Combination", yaxis_title="Number of Duplicate Code Samples",
                          width=800, height=500, showlegend=False, plot_bgcolor='white')
        fig.update_yaxes(gridcolor='lightgray', gridwidth=1)
        fig.update_xaxes(gridcolor='lightgray', gridwidth=1)
        total_duplicates = sum(overlap_details.values())
        for i, (split_combo, count) in enumerate(overlap_details.items()):
            percentage = (count / total_duplicates * 100) if total_duplicates > 0 else 0
            fig.add_annotation(x=split_combo, y=count + max(overlap_details.values()) * 0.02,
                               text=f"{percentage:.1f}%", showarrow=False, font=dict(size=12, color="black"))
        if save_path:
            self._safe_save_figure(fig, save_path, "chart14_simple_data_leakage_breakdown")
        fig.show()
        return fig
    
    def generate_all_charts(self, save_path=None):
        if save_path:
            Path(save_path).mkdir(parents=True, exist_ok=True)
        charts = {}
        charts['split_proportions'] = self.chart1_split_proportions(save_path)
        charts['language_distribution'] = self.chart2_language_distribution(save_path)
        charts['target_distribution'] = self.chart3_target_distribution(save_path)
        charts['language_target_heatmap'] = self.chart4_language_target_heatmap(save_path)
        charts['code_length_distribution'] = self.chart5_code_length_distribution(save_path)
        charts['model_distribution'] = self.chart6_model_distribution(save_path)
        charts['model_split_distribution'] = self.chart7_model_split_distribution(save_path)
        charts['target_balance'] = self.chart8_target_balance_analysis(save_path)
        charts['code_length_by_language'] = self.chart9_code_length_by_language(save_path)
        charts['language_split_sunburst'] = self.chart10_language_split_sunburst(save_path)
        charts['summary_table'], summary_df = self.chart11_summary_statistics_table(save_path)
        charts['correlation_matrix'] = self.chart12_correlation_matrix(save_path)
        leakage_result = self.chart13_data_leakage_analysis(save_path)
        charts['data_leakage_analysis'] = leakage_result[0] if leakage_result else None
        charts['simple_data_leakage'] = self.chart14_simple_data_leakage_breakdown(save_path)
        if save_path:
            summary_df.to_csv(f"{save_path}/dataset_summary_statistics.csv", index=False)
            if leakage_result:
                leakage_df = pd.DataFrame([leakage_result[1]['summary_stats']])
                leakage_df.to_csv(f"{save_path}/data_leakage_report.csv", index=False)
        return charts

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-path', default='data/')
    parser.add_argument('--save-path', default=None)
    parser.add_argument('--charts', nargs='+', choices=['all', '1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13', '14'], default=['all'])
    args = parser.parse_args()
    analyzer = CoDeTDataAnalyzer(data_path=args.data_path)
    analyzer.load_data()
    if 'all' in args.charts:
        analyzer.generate_all_charts(save_path=args.save_path)
    else:
        chart_methods = {
            '1': analyzer.chart1_split_proportions, '2': analyzer.chart2_language_distribution,
            '3': analyzer.chart3_target_distribution, '4': analyzer.chart4_language_target_heatmap,
            '5': analyzer.chart5_code_length_distribution, '6': analyzer.chart6_model_distribution,
            '7': analyzer.chart7_model_split_distribution, '8': analyzer.chart8_target_balance_analysis,
            '9': analyzer.chart9_code_length_by_language, '10': analyzer.chart10_language_split_sunburst,
            '11': analyzer.chart11_summary_statistics_table, '12': analyzer.chart12_correlation_matrix,
            '13': analyzer.chart13_data_leakage_analysis, '14': analyzer.chart14_simple_data_leakage_breakdown
        }
        for chart_num in args.charts:
            chart_methods[chart_num](save_path=args.save_path)

if __name__ == "__main__":
    main()