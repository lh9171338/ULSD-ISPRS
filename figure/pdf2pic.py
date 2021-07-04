import argparse
import fitz


def pic2pdf(pic_file, pdf_file):
    doc = fitz.open()
    imgdoc = fitz.open(pic_file)
    pdfbytes = imgdoc.convertToPDF()
    imgpdf = fitz.open('pdf', pdfbytes)
    doc.insertPDF(imgpdf)
    doc.save(pdf_file)
    doc.close()


def pdf2pic(pdf_file, pic_file, zoom=None):
    if zoom is None:
        zoom = 1
    pdf = fitz.open(pdf_file)
    page = pdf[0]
    trans = fitz.Matrix(zoom, zoom)
    pm = page.getPixmap(matrix=trans)
    pm.writeImage(pic_file)
    pdf.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input', type=str, help='filename of input pdf or image file', required=True)
    parser.add_argument('-o', '--output', type=str, help='filename of output image or pdf file')
    parser.add_argument('-z', '--zoom', type=float, help='zoom image')
    opts = parser.parse_args()

    postfix = opts.input.split('.')[-1]
    if postfix == 'pdf':
        opts.output = '.'.join(opts.input.split('.')[:-1]) + '.png' if opts.output is None else opts.output
        print(opts)

        pdf_file = opts.input
        pic_file = opts.output
        zoom = opts.zoom
        pdf2pic(pdf_file, pic_file, zoom)
    else:
        opts.output = '.'.join(opts.input.split('.')[:-1]) + '.pdf' if opts.output is None else opts.output
        print(opts)

        pic_file = opts.input
        pdf_file = opts.output
        pic2pdf(pic_file, pdf_file)
