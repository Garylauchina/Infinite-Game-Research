#!/usr/bin/env python3
"""
Convert Markdown to PDF using markdown and reportlab
"""

import sys
import os
from pathlib import Path

try:
    import markdown
    from reportlab.lib.pagesizes import letter, A4
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.units import inch
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, PageBreak
    from reportlab.lib.enums import TA_LEFT, TA_CENTER
    from reportlab.pdfbase import pdfmetrics
    from reportlab.pdfbase.ttfonts import TTFont
    from reportlab.lib.colors import black, HexColor
except ImportError as e:
    print(f"Error: Missing required library: {e}")
    print("\nPlease install required libraries:")
    print("  pip install markdown reportlab")
    sys.exit(1)

def markdown_to_pdf(md_file, pdf_file):
    """Convert Markdown file to PDF"""
    
    # Read Markdown file
    with open(md_file, 'r', encoding='utf-8') as f:
        md_content = f.read()
    
    # Convert Markdown to HTML
    html = markdown.markdown(md_content, extensions=['extra', 'codehilite', 'tables'])
    
    # Create PDF
    doc = SimpleDocTemplate(
        pdf_file,
        pagesize=A4,
        rightMargin=72,
        leftMargin=72,
        topMargin=72,
        bottomMargin=72
    )
    
    # Styles
    styles = getSampleStyleSheet()
    
    # Custom styles
    title_style = ParagraphStyle(
        'CustomTitle',
        parent=styles['Heading1'],
        fontSize=18,
        textColor=black,
        spaceAfter=12,
        alignment=TA_LEFT
    )
    
    heading1_style = ParagraphStyle(
        'CustomHeading1',
        parent=styles['Heading1'],
        fontSize=14,
        textColor=black,
        spaceAfter=12,
        spaceBefore=12,
        alignment=TA_LEFT
    )
    
    heading2_style = ParagraphStyle(
        'CustomHeading2',
        parent=styles['Heading2'],
        fontSize=12,
        textColor=black,
        spaceAfter=8,
        spaceBefore=8,
        alignment=TA_LEFT
    )
    
    normal_style = ParagraphStyle(
        'CustomNormal',
        parent=styles['Normal'],
        fontSize=10,
        textColor=black,
        spaceAfter=6,
        alignment=TA_LEFT
    )
    
    # Parse HTML and build story
    story = []
    
    # Simple HTML parser (basic implementation)
    import re
    from html.parser import HTMLParser
    
    class SimpleHTMLParser(HTMLParser):
        def __init__(self, story, styles):
            super().__init__()
            self.story = story
            self.styles = styles
            self.current_text = []
            self.in_code = False
            self.in_pre = False
            
        def handle_starttag(self, tag, attrs):
            if tag == 'h1':
                if self.current_text:
                    text = ''.join(self.current_text).strip()
                    if text:
                        self.story.append(Paragraph(text, self.styles['CustomTitle']))
                        self.story.append(Spacer(1, 0.2*inch))
                    self.current_text = []
            elif tag == 'h2':
                if self.current_text:
                    text = ''.join(self.current_text).strip()
                    if text:
                        self.story.append(Paragraph(text, self.styles['CustomHeading1']))
                        self.story.append(Spacer(1, 0.1*inch))
                    self.current_text = []
            elif tag == 'h3':
                if self.current_text:
                    text = ''.join(self.current_text).strip()
                    if text:
                        self.story.append(Paragraph(text, self.styles['CustomHeading2']))
                        self.story.append(Spacer(1, 0.05*inch))
                    self.current_text = []
            elif tag in ['code', 'pre']:
                self.in_code = True
            elif tag == 'p':
                if self.current_text:
                    text = ''.join(self.current_text).strip()
                    if text:
                        self.story.append(Paragraph(text, self.styles['CustomNormal']))
                        self.story.append(Spacer(1, 0.05*inch))
                    self.current_text = []
            elif tag == 'br':
                self.current_text.append('\n')
                
        def handle_endtag(self, tag):
            if tag in ['code', 'pre']:
                self.in_code = False
            elif tag == 'p':
                if self.current_text:
                    text = ''.join(self.current_text).strip()
                    if text:
                        self.story.append(Paragraph(text, self.styles['CustomNormal']))
                        self.story.append(Spacer(1, 0.05*inch))
                    self.current_text = []
                    
        def handle_data(self, data):
            # Clean up data
            data = data.replace('&nbsp;', ' ')
            data = data.replace('&lt;', '<')
            data = data.replace('&gt;', '>')
            data = data.replace('&amp;', '&')
            self.current_text.append(data)
    
    # Convert HTML to story
    parser = SimpleHTMLParser(story, {
        'CustomTitle': title_style,
        'CustomHeading1': heading1_style,
        'CustomHeading2': heading2_style,
        'CustomNormal': normal_style
    })
    
    # Clean HTML for parsing
    html_clean = re.sub(r'<script.*?</script>', '', html, flags=re.DOTALL)
    html_clean = re.sub(r'<style.*?</style>', '', html_clean, flags=re.DOTALL)
    
    parser.feed(html_clean)
    
    # Add any remaining text
    if parser.current_text:
        text = ''.join(parser.current_text).strip()
        if text:
            story.append(Paragraph(text, normal_style))
    
    # Build PDF
    doc.build(story)
    print(f"✓ PDF created: {pdf_file}")

if __name__ == '__main__':
    md_file = 'InfiniteGame_V5_TechnicalNote.md'
    pdf_file = 'InfiniteGame_V5_TechnicalNote.pdf'
    
    if not os.path.exists(md_file):
        print(f"Error: {md_file} not found")
        sys.exit(1)
    
    markdown_to_pdf(md_file, pdf_file)
