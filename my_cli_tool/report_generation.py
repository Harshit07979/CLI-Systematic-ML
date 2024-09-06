from fpdf import FPDF

def generate_report(evaluation_results, report_format='text', model_params=None):
    print(f"Generating report in {report_format} format...")

    try:
        if report_format == 'text':
            with open('reports/model_report.txt', 'w') as f:
                f.write("### Model Performance Report\n\n")
                for metric, value in evaluation_results.items():
                    f.write(f"{metric.capitalize()}: {value:.2f}\n")
                print("Text report generated: reports/model_report.txt")

        elif report_format == 'html':
            with open('reports/model_report.html', 'w') as f:
                f.write("<h2>Model Performance Report</h2>\n")
                for metric, value in evaluation_results.items():
                    f.write(f"<p><b>{metric.capitalize()}:</b> {value:.2f}</p>\n")
                print("HTML report generated: reports/model_report.html")

        elif report_format == 'pdf':
            pdf = FPDF()
            pdf.add_page()
            pdf.set_font("Arial", size=12)
            pdf.cell(200, 10, txt="Model Performance Report", ln=True, align='C')

            for metric, value in evaluation_results.items():
                pdf.cell(200, 10, txt=f"{metric.capitalize()}: {value:.2f}", ln=True)

            pdf.output("reports/model_report.pdf")
            print("PDF report generated: reports/model_report.pdf")

        else:
            print(f"Error: Unsupported report format '{report_format}'")

    except Exception as e:
        print(f"Error generating report: {e}")
