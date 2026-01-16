#!/bin/bash
# 简单的转换脚本 - 使用 macOS 内置工具

MD_FILE="InfiniteGame_V5_TechnicalNote.md"
HTML_FILE="${MD_FILE%.md}.html"
PDF_FILE="${MD_FILE%.md}.pdf"

# 检查是否有 markdown 命令（macOS 可能没有）
if command -v markdown &> /dev/null; then
    markdown "$MD_FILE" > "$HTML_FILE"
    echo "HTML created: $HTML_FILE"
    echo "You can now open $HTML_FILE in a browser and print to PDF"
else
    echo "markdown command not found"
    echo ""
    echo "Please use one of these methods:"
    echo "1. Install pandoc: brew install pandoc"
    echo "2. Use online tool: https://www.markdowntopdf.com/"
    echo "3. Use VS Code extension: Markdown PDF"
    echo ""
    echo "See CONVERT_TO_PDF.md for detailed instructions"
fi
