"""Step 18: copy the rendered figures to figures_for_submission/ under the names used in the article."""
import shutil, os
MAP = {'fig1': 'Figure_1', 'fig6': 'Figure_2', 'fig_scale': 'Figure_3', 'fig3': 'Figure_4', 'fig4': 'Figure_5', 'fig7': 'Figure_6',
       'fig_robust': 'Figure_7', 'figGA': 'Graphical_abstract', 'fig2': 'Figure_S1', 'fig5': 'Figure_S2',
       'fig8': 'Figure_S3', 'figB1': 'Figure_S4', 'fig10': 'Figure_S5'}
os.makedirs('figures_for_submission', exist_ok=True)
for src, dst in {**MAP, 'dib_fig1': 'DiB_Figure_1', 'dib_fig2': 'DiB_Figure_2', 'dib_fig3': 'DiB_Figure_3', 'dib_fig4': 'DiB_Figure_4'}.items():
    for ext in ('png', 'pdf'):
        if os.path.exists(f'{src}.{ext}'): shutil.copy(f'{src}.{ext}', f'figures_for_submission/{dst}.{ext}')
print('figures collected:', sorted(os.listdir('figures_for_submission')))
