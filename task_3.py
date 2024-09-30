import os

import numpy as np
import pandas as pd
import plotly.express as px
from matplotlib import pyplot as plt
from plotly.offline import plot
import time
import plotly.io as pio
import matplotlib.backends.backend_pdf as mpdf
from matplotlib.backends.backend_pdf import PdfPages

def feature_evaluation(X: pd.DataFrame, y: pd.Series, output_path: str = "."):
    corr_arr = []
    for ft in X:
        # calc pearson correlation
        cov = np.cov(X[ft], y)[0, 1]
        std1 = np.std(X[ft])
        std2 = np.std(y)
        corr = cov / (std1 * std2)
        corr_arr.append((ft, corr))

    sorted_data = sorted(corr_arr, key=lambda x: x[1], reverse=True)

    # Retrieve the top six entries with the highest correlation
    top_six_entries = sorted_data[:6]

    pdf_filename = 'agoda_churn_prediction_model.pdf'

    # Create a PDF document
    pdf = PdfPages(pdf_filename)

    # Create and save scatter plots
    for i, (ft, corr) in enumerate(top_six_entries):
        fig = px.scatter(pd.DataFrame({'x': X[ft], 'cancellation': y}), x='x', y='cancellation', trendline='ols',
                         title=f'Scatter plot of {ft} vs cancellation (Pearson corr = {corr:.2f})')
        fig.update_layout(showlegend=False)

        # Save the Plotly figure as a PNG image
        image_file = f'scatter_plot_{ft}.png'
        fig.write_image(image_file)

        # Convert the PNG image to PDF and append it to the PDF document
        image = plt.imread(image_file)
        fig, ax = plt.subplots(figsize=(image.shape[1] / 80, image.shape[0] / 80))
        ax.imshow(image)
        ax.axis('off')
        pdf.savefig(fig)

        # Delete the image file
        os.remove(image_file)

    # Close the PDF document
    pdf.close()