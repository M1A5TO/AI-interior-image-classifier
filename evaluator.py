import json
import pandas as pd
import numpy as np
from collections import defaultdict
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.patches import FancyBboxPatch
import matplotlib.gridspec as gridspec


class VisualEffectivenessEvaluator:
    def __init__(self, ground_truth_file=None):
        """
        Inicjalizacja evaluatora z zaawansowanymi wizualizacjami
        """
        self.ground_truth = self._load_ground_truth(ground_truth_file) if ground_truth_file else {}
        self.results = {}
        self.categories = ['room_types', 'styles', 'characteristics', 'materials', 'colors']

        # Konfiguracja stylu matplotlib
        plt.style.use('default')
        self.colors = {
            'success': '#2ecc71',
            'failure': '#e74c3c',
            'neutral': '#3498db',
            'correct': '#27ae60',
            'incorrect': '#c0392b',
            'background': '#f8f9fa'
        }

    def _load_ground_truth(self, file_path):
        """Ładuje prawdziwe etykiety z pliku JSON/CSV"""
        if file_path.endswith('.json'):
            with open(file_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        elif file_path.endswith('.csv'):
            df = pd.read_csv(file_path)
            return df.set_index('image_id').to_dict('index')
        else:
            print("Nieobsługiwany format pliku. Użyj JSON lub CSV.")
            return {}

    def load_analysis_results(self, results_file):
        """Ładuje wyniki analizy modelu"""
        try:
            with open(results_file, 'r', encoding='utf-8') as f:
                self.results = json.load(f)
            print(f"Załadowano wyniki dla {len(self.results)} obrazów")
            return True
        except Exception as e:
            print(f"Błąd przy ładowaniu wyników: {e}")
            return False

    def evaluate_single_image(self, image_id, analysis_result, ground_truth=None):
        """
        Ewaluacja pojedynczego obrazu
        """
        if ground_truth is None:
            ground_truth = self.ground_truth.get(image_id, {})

        metrics = {
            'image_id': image_id,
            'correct_categories': 0,
            'total_categories_checked': 0,
            'category_details': {},
            'success': False,
            'missing_categories': [],
            'confidences': []
        }

        if 'analysis' not in analysis_result or not analysis_result['analysis']:
            return metrics

        analysis = analysis_result['analysis']

        for category in self.categories:
            if category not in analysis or not analysis[category]:
                continue

            if category not in ground_truth or not ground_truth[category]:
                metrics['missing_categories'].append(category)
                continue

            metrics['total_categories_checked'] += 1
            top_prediction = analysis[category][0]
            predicted_attribute, confidence = top_prediction

            metrics['confidences'].append(confidence)

            true_attributes = ground_truth[category]
            if not isinstance(true_attributes, list):
                true_attributes = [true_attributes]

            is_correct = predicted_attribute in true_attributes

            metrics['category_details'][category] = {
                'predicted': predicted_attribute,
                'true': true_attributes,
                'confidence': confidence,
                'correct': is_correct
            }

            if is_correct:
                metrics['correct_categories'] += 1

        metrics['success'] = metrics['correct_categories'] >= 3
        metrics['avg_confidence'] = np.mean(metrics['confidences']) if metrics['confidences'] else 0

        return metrics

    def evaluate_all_images(self):
        """Ewaluuje wszystkie załadowane obrazy"""
        if not self.results:
            print("Brak wyników do ewaluacji")
            return {}

        if not self.ground_truth:
            print("UWAGA: Brak ground truth. Ewaluacja wymaga pliku z prawdziwymi etykietami.")
            return {}

        evaluation_results = {}
        summary = {
            'total_images': 0,
            'successful_images': 0,
            'failed_images': 0,
            'avg_correct_categories': 0,
            'success_rate': 0,
            'category_accuracy': {category: {'correct': 0, 'total': 0} for category in self.categories},
            'confidence_stats': {category: [] for category in self.categories},
            'all_confidences': [],
            'all_correct_counts': []
        }

        for image_id, analysis_result in self.results.items():
            if image_id not in self.ground_truth:
                continue

            metrics = self.evaluate_single_image(image_id, analysis_result)
            evaluation_results[image_id] = metrics

            summary['total_images'] += 1
            summary['avg_correct_categories'] += metrics['correct_categories']
            summary['all_correct_counts'].append(metrics['correct_categories'])
            summary['all_confidences'].extend(metrics['confidences'])

            if metrics['success']:
                summary['successful_images'] += 1
            else:
                summary['failed_images'] += 1

            for category, details in metrics['category_details'].items():
                summary['category_accuracy'][category]['total'] += 1
                if details['correct']:
                    summary['category_accuracy'][category]['correct'] += 1
                summary['confidence_stats'][category].append(details['confidence'])

        if summary['total_images'] > 0:
            summary['avg_correct_categories'] /= summary['total_images']
            summary['success_rate'] = (summary['successful_images'] / summary['total_images']) * 100

            for category in self.categories:
                correct = summary['category_accuracy'][category]['correct']
                total = summary['category_accuracy'][category]['total']
                if total > 0:
                    summary['category_accuracy'][category]['accuracy'] = (correct / total) * 100
                else:
                    summary['category_accuracy'][category]['accuracy'] = 0

                if summary['confidence_stats'][category]:
                    summary['category_accuracy'][category]['avg_confidence'] = np.mean(
                        summary['confidence_stats'][category])
                else:
                    summary['category_accuracy'][category]['avg_confidence'] = 0

        return evaluation_results, summary

    def create_comprehensive_dashboard(self, evaluation_results, summary):
        """Tworzy kompleksowy dashboard z wizualizacjami"""

        # Przygotowanie danych
        correct_counts = summary['all_correct_counts']
        success_status = [1 if metrics['success'] else 0 for metrics in evaluation_results.values()]
        categories = list(self.categories)
        accuracies = [summary['category_accuracy'][cat]['accuracy'] for cat in categories]
        avg_confidences = [summary['category_accuracy'][cat].get('avg_confidence', 0) for cat in categories]

        # Tworzenie dashboardu
        fig = plt.figure(figsize=(20, 16))
        fig.suptitle('DASHBOARD ANALIZY SKUTECZNOŚCI MODELU\nKryterium: ≥3 z 5 kategorii poprawnych',
                     fontsize=16, fontweight='bold', y=0.98)

        # Definicja siatki wykresów
        gs = gridspec.GridSpec(3, 3, figure=fig, hspace=0.4, wspace=0.3)

        # Wykres 1: Rozkład poprawnych kategorii (górny lewy)
        ax1 = fig.add_subplot(gs[0, 0])
        self._plot_category_distribution(ax1, correct_counts, summary)

        # Wykres 2: Skuteczność per kategoria (górny środkowy)
        ax2 = fig.add_subplot(gs[0, 1])
        self._plot_category_accuracy(ax2, categories, accuracies)

        # Wykres 3: Wskaźnik sukcesu (górny prawy)
        ax3 = fig.add_subplot(gs[0, 2])
        self._plot_success_rate(ax3, summary)

        # Wykres 4: Pewność vs dokładność (środkowy lewy)
        ax4 = fig.add_subplot(gs[1, 0])
        self._plot_confidence_vs_accuracy(ax4, categories, accuracies, avg_confidences)

        # Wykres 5: Heatmapa korelacji (środkowy środkowy)
        ax5 = fig.add_subplot(gs[1, 1])
        self._plot_correlation_heatmap(ax5, evaluation_results, categories)

        # Wykres 6: Skumulowana skuteczność (środkowy prawy)
        ax6 = fig.add_subplot(gs[1, 2])
        self._plot_cumulative_success(ax6, correct_counts)

        # Wykres 7: Analiza porażek (dolny lewy)
        ax7 = fig.add_subplot(gs[2, 0])
        self._plot_failure_analysis(ax7, evaluation_results)

        # Wykres 8: Top poprawne/niepoprawne predykcje (dolny środkowy)
        ax8 = fig.add_subplot(gs[2, 1])
        self._plot_top_predictions(ax8, evaluation_results)

        # Wykres 9: Podsumowanie metryk (dolny prawy)
        ax9 = fig.add_subplot(gs[2, 2])
        self._plot_metrics_summary(ax9, summary)

        plt.tight_layout()
        plt.savefig('comprehensive_dashboard.png', dpi=300, bbox_inches='tight',
                    facecolor=self.colors['background'])
        plt.show()

        print("Kompleksowy dashboard zapisano do: comprehensive_dashboard.png")

    def _plot_category_distribution(self, ax, correct_counts, summary):
        """Wykres rozkładu liczby poprawnych kategorii"""
        counts, bins, patches = ax.hist(correct_counts, bins=range(0, 7),
                                        alpha=0.8, color=self.colors['neutral'],
                                        edgecolor='white', linewidth=2, align='left')

        # Kolorowanie słupków
        for i, (count, patch) in enumerate(zip(counts, patches)):
            if i >= 3:
                patch.set_facecolor(self.colors['success'])
            else:
                patch.set_facecolor(self.colors['failure'])

        ax.set_xlabel('Liczba poprawnych kategorii', fontweight='bold')
        ax.set_ylabel('Liczba obrazów', fontweight='bold')
        ax.set_title('ROZKŁAD POPRAWNYCH KATEGORII', fontweight='bold', pad=20)
        ax.set_xticks(range(0, 6))
        ax.grid(True, alpha=0.3, axis='y')
        ax.axvline(x=2.5, color='red', linestyle='--', linewidth=2,
                   label=f'Próg sukcesu (≥3)\nSkuteczność: {summary["success_rate"]:.1f}%')
        ax.legend()

        # Dodanie wartości na słupkach
        for i, count in enumerate(counts):
            if count > 0:
                ax.text(i, count + 0.1, f'{int(count)}', ha='center', va='bottom',
                        fontweight='bold', fontsize=10)

    def _plot_category_accuracy(self, ax, categories, accuracies):
        """Wykres dokładności per kategoria"""
        y_pos = np.arange(len(categories))
        bars = ax.barh(y_pos, accuracies, color=[self.colors['success'] if acc >= 50 else self.colors['failure']
                                                 for acc in accuracies], alpha=0.8)

        ax.set_yticks(y_pos)
        ax.set_yticklabels([cat.replace('_', ' ').title() for cat in categories])
        ax.set_xlabel('Dokładność (%)', fontweight='bold')
        ax.set_title('DOKŁADNOŚĆ PER KATEGORIA', fontweight='bold', pad=20)
        ax.set_xlim(0, 100)
        ax.grid(True, alpha=0.3, axis='x')
        ax.axvline(x=50, color='red', linestyle='--', alpha=0.7, label='Próg 50%')

        # Dodanie wartości na słupkach
        for i, (bar, acc) in enumerate(zip(bars, accuracies)):
            ax.text(bar.get_width() + 1, bar.get_y() + bar.get_height() / 2,
                    f'{acc:.1f}%', va='center', fontweight='bold',
                    color='green' if acc >= 50 else 'red')

    def _plot_success_rate(self, ax, summary):
        """Wykres kołowy wskaźnika sukcesu"""
        sizes = [summary['successful_images'], summary['failed_images']]
        labels = [f'Sukces\n{summary["successful_images"]} img\n({summary["success_rate"]:.1f}%)',
                  f'Porażka\n{summary["failed_images"]} img\n({100 - summary["success_rate"]:.1f}%)']
        colors = [self.colors['success'], self.colors['failure']]
        explode = (0.1, 0)

        wedges, texts, autotexts = ax.pie(sizes, explode=explode, labels=labels, colors=colors,
                                          autopct='', startangle=90, shadow=True)

        ax.set_title('WSKAŹNIK SUKCESU', fontweight='bold', pad=20)

        # Stylizacja tekstu
        for text in texts:
            text.set_fontweight('bold')
            text.set_fontsize(10)

    def _plot_confidence_vs_accuracy(self, ax, categories, accuracies, avg_confidences):
        """Wykres pewności vs dokładność"""
        scatter = ax.scatter(avg_confidences, accuracies, s=200, alpha=0.7,
                             c=[self.colors['success'] if acc >= 50 else self.colors['failure']
                                for acc in accuracies],
                             edgecolors='black', linewidths=1)

        ax.set_xlabel('Średnia pewność (%)', fontweight='bold')
        ax.set_ylabel('Dokładność (%)', fontweight='bold')
        ax.set_title('PEWNOŚĆ vs DOKŁADNOŚĆ', fontweight='bold', pad=20)
        ax.grid(True, alpha=0.3)
        ax.set_xlim(0, 100)
        ax.set_ylim(0, 100)

        # Dodanie linii referencyjnych
        ax.axhline(y=50, color='red', linestyle='--', alpha=0.5)
        ax.axvline(x=50, color='red', linestyle='--', alpha=0.5)

        # Dodanie etykiet punktów
        for i, category in enumerate(categories):
            ax.annotate(category.replace('_', '\n').title(),
                        (avg_confidences[i], accuracies[i]),
                        xytext=(10, 10), textcoords='offset points',
                        fontsize=8, fontweight='bold',
                        bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.7))

    def _plot_correlation_heatmap(self, ax, evaluation_results, categories):
        """Heatmapa korelacji między kategoriami"""
        # Przygotowanie danych do heatmapy
        correlation_data = []
        for image_id, metrics in evaluation_results.items():
            row = []
            for category in categories:
                if category in metrics['category_details']:
                    row.append(1 if metrics['category_details'][category]['correct'] else 0)
                else:
                    row.append(np.nan)
            correlation_data.append(row)

        if correlation_data:
            df = pd.DataFrame(correlation_data, columns=categories)
            correlation_matrix = df.corr()

            mask = np.triu(np.ones_like(correlation_matrix, dtype=bool))
            sns.heatmap(correlation_matrix, mask=mask, annot=True, cmap='RdYlGn',
                        center=0, ax=ax, square=True, fmt='.2f',
                        cbar_kws={'shrink': 0.8})

            ax.set_title('KORELACJA MIĘDZY KATEGORIAMI\n(czy sukces w jednej kategorii\nwpływa na inne?)',
                         fontweight='bold', pad=20)
            ax.set_xticklabels([cat.replace('_', '\n').title() for cat in categories],
                               rotation=45, ha='right')
            ax.set_yticklabels([cat.replace('_', '\n').title() for cat in categories],
                               rotation=0)

    def _plot_cumulative_success(self, ax, correct_counts):
        """Wykres skumulowanej skuteczności"""
        sorted_counts = np.sort(correct_counts)
        cumulative = np.cumsum(sorted_counts) / np.sum(sorted_counts) * 100

        ax.plot(sorted_counts, cumulative, linewidth=3, marker='o',
                color=self.colors['neutral'], markersize=6)
        ax.fill_between(sorted_counts, cumulative, alpha=0.3, color=self.colors['neutral'])

        ax.set_xlabel('Liczba poprawnych kategorii', fontweight='bold')
        ax.set_ylabel('Skumulowany % obrazów', fontweight='bold')
        ax.set_title('SKUMULOWANA SKUTECZNOŚĆ', fontweight='bold', pad=20)
        ax.grid(True, alpha=0.3)
        ax.set_xticks(range(0, 6))

        # Zaznaczenie progu 3 kategorii
        threshold_idx = np.where(sorted_counts >= 3)[0]
        if len(threshold_idx) > 0:
            threshold_x = 3
            threshold_y = cumulative[threshold_idx[0]]
            ax.axvline(x=threshold_x, color='red', linestyle='--', alpha=0.7)
            ax.axhline(y=threshold_y, color='red', linestyle='--', alpha=0.7)
            ax.plot(threshold_x, threshold_y, 'ro', markersize=10)
            ax.text(threshold_x, threshold_y + 5, f'{threshold_y:.1f}% obrazów\n≥3 kategorie',
                    ha='center', fontweight='bold', bbox=dict(boxstyle="round,pad=0.3", fc="white"))

    def _plot_failure_analysis(self, ax, evaluation_results):
        """Analiza przyczyn porażek"""
        failures = {img_id: metrics for img_id, metrics in evaluation_results.items()
                    if not metrics['success']}

        if not failures:
            ax.text(0.5, 0.5, 'BRAK PORAZEK!', ha='center', va='center',
                    fontsize=16, fontweight='bold', color='green',
                    transform=ax.transAxes)
            ax.set_title('ANALIZA PORAZEK', fontweight='bold', pad=20)
            return

        failure_reasons = defaultdict(int)
        for metrics in failures.values():
            failure_reasons[metrics['correct_categories']] += 1

        reasons = list(failure_reasons.keys())
        counts = list(failure_reasons.values())

        bars = ax.bar(reasons, counts, color=self.colors['failure'], alpha=0.8)
        ax.set_xlabel('Liczba poprawnych kategorii', fontweight='bold')
        ax.set_ylabel('Liczba obrazów', fontweight='bold')
        ax.set_title('ANALIZA PORAZEK\n(dlaczego obrazy nie spełniają kryterium?)',
                     fontweight='bold', pad=20)
        ax.set_xticks(reasons)
        ax.grid(True, alpha=0.3, axis='y')

        # Dodanie wartości na słupkach
        for bar, count in zip(bars, counts):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.1,
                    f'{count}', ha='center', va='bottom', fontweight='bold')

    def _plot_top_predictions(self, ax, evaluation_results):
        """Top poprawne i niepoprawne predykcje"""
        all_predictions = []
        for metrics in evaluation_results.values():
            for category, details in metrics['category_details'].items():
                all_predictions.append({
                    'category': category,
                    'predicted': details['predicted'],
                    'correct': details['correct'],
                    'confidence': details['confidence']
                })

        df = pd.DataFrame(all_predictions)

        if len(df) > 0:
            # Top poprawne predykcje
            correct_top = df[df['correct']].groupby('predicted').size().nlargest(5)
            # Top niepoprawne predykcje
            incorrect_top = df[~df['correct']].groupby('predicted').size().nlargest(5)

            # Wykres podwójny
            y_pos = np.arange(max(len(correct_top), len(incorrect_top)))

            ax.barh(y_pos - 0.2, correct_top.values, 0.4, label='Poprawne',
                    color=self.colors['correct'], alpha=0.8)
            ax.barh(y_pos + 0.2, incorrect_top.values, 0.4, label='Niepoprawne',
                    color=self.colors['incorrect'], alpha=0.8)

            ax.set_yticks(y_pos)
            ax.set_yticklabels(correct_top.index if len(correct_top) >= len(incorrect_top) else incorrect_top.index)
            ax.set_xlabel('Liczba wystąpień', fontweight='bold')
            ax.set_title('TOP PRZEWIDYWANIA\n(5 najczęstszych)', fontweight='bold', pad=20)
            ax.legend()
            ax.grid(True, alpha=0.3, axis='x')

    def _plot_metrics_summary(self, ax, summary):
        """Podsumowanie kluczowych metryk"""
        metrics_data = [
            ('Całkowita liczba\nobrazów', summary['total_images'], ''),
            ('Skuteczność', summary['success_rate'], '%'),
            ('Średnia poprawnych\nkategorii', f'{summary["avg_correct_categories"]:.2f}', '/5'),
            ('Obrazy z ≥3 kat.', f'{summary["successful_images"]}/{summary["total_images"]}', ''),
            ('Średnia pewność', f'{np.mean(summary["all_confidences"]):.1f}', '%')
        ]

        names = [m[0] for m in metrics_data]
        values = [m[1] for m in metrics_data]
        units = [m[2] for m in metrics_data]

        y_pos = np.arange(len(metrics_data))

        bars = ax.barh(y_pos, [100 if '%' in str(v) else v if isinstance(v, (int, float)) else 0
                               for v in values],
                       color=self.colors['neutral'], alpha=0.7)

        ax.set_yticks(y_pos)
        ax.set_yticklabels(names)
        ax.set_xlim(0, 110)
        ax.set_title('PODSUMOWANIE METRYK', fontweight='bold', pad=20)
        ax.grid(True, alpha=0.3, axis='x')

        # Dodanie wartości metryk
        for i, (bar, value, unit) in enumerate(zip(bars, values, units)):
            ax.text(bar.get_width() + 2, bar.get_y() + bar.get_height() / 2,
                    f'{value}{unit}', va='center', fontweight='bold', fontsize=10)

    def generate_detailed_report(self, evaluation_results, summary):
        """Generuje szczegółowy raport z wizualizacjami"""
        print("=" * 80)
        print("KOMPLEKSOWY RAPORT SKUTECZNOŚCI MODELU")
        print("=" * 80)
        print("KRYTERIUM: co najmniej 3 z 5 kategorii prawidłowo się pokrywa")
        print("=" * 80)

        print(f"\n📊 PODSUMOWANIE:")
        print(f"   Łączna liczba obrazów: {summary['total_images']}")
        print(f"   ✅ Poprawne analizy (≥3 kategorie): {summary['successful_images']}")
        print(f"   ❌ Niepoprawne analizy: {summary['failed_images']}")
        print(f"   🎯 WSKAŹNIK SUKCESU: {summary['success_rate']:.2f}%")
        print(f"   📈 Średnia poprawnych kategorii: {summary['avg_correct_categories']:.2f}/5")

        # Tworzenie dashboardu
        self.create_comprehensive_dashboard(evaluation_results, summary)

        return summary


# Przykład użycia
def main():
    import argparse

    parser = argparse.ArgumentParser(description='Evaluator z zaawansowanymi wizualizacjami')
    parser.add_argument('--results', type=str, required=True, help='Plik z wynikami analizy modelu (JSON)')
    parser.add_argument('--ground-truth', type=str, required=True, help='Plik z prawdziwymi etykietami (JSON/CSV)')

    args = parser.parse_args()

    # Inicjalizacja evaluatora
    evaluator = VisualEffectivenessEvaluator(args.ground_truth)

    # Ładowanie wyników
    if not evaluator.load_analysis_results(args.results):
        exit(1)

    # Ewaluacja
    print("Przeprowadzanie ewaluacji z zaawansowanymi wizualizacjami...")
    evaluation_results, summary = evaluator.evaluate_all_images()

    # Generowanie raportu i dashboardu
    evaluator.generate_detailed_report(evaluation_results, summary)

    print("\n" + "=" * 80)
    print("EWALUACJA ZAKOŃCZONA")
    print("Dashboard zapisano jako: comprehensive_dashboard.png")
    print("=" * 80)


if __name__ == "__main__":
    main()