# 将 Markdown 转换为 PDF 的方法

## 方法 1: 使用 Pandoc（推荐）

### 安装 Pandoc

**macOS (使用 Homebrew)**:
```bash
brew install pandoc
brew install basictex  # 或 mactex（完整版，较大）
```

**或使用 MacTeX**:
```bash
brew install --cask mactex
```

### 转换命令

```bash
pandoc InfiniteGame_V5_TechnicalNote.md -o InfiniteGame_V5_TechnicalNote.pdf \
  --pdf-engine=xelatex \
  --template=default \
  -V geometry:margin=1in \
  -V fontsize=11pt \
  --toc
```

**简化版（如果 xelatex 不可用）**:
```bash
pandoc InfiniteGame_V5_TechnicalNote.md -o InfiniteGame_V5_TechnicalNote.pdf \
  -V geometry:margin=1in \
  -V fontsize=11pt
```

## 方法 2: 使用在线工具

1. **Markdown to PDF** (https://www.markdowntopdf.com/)
   - 上传 `InfiniteGame_V5_TechnicalNote.md`
   - 下载生成的 PDF

2. **Dillinger** (https://dillinger.io/)
   - 打开文件
   - 点击 "Export as" → "PDF"

3. **GitPrint** (https://gitprint.com/)
   - 如果文件在 GitHub 上，可以直接打印为 PDF

## 方法 3: 使用 VS Code 扩展

1. 安装扩展 "Markdown PDF"
2. 打开 `InfiniteGame_V5_TechnicalNote.md`
3. 右键 → "Markdown PDF: Export (pdf)"

## 方法 4: 使用 Python（需要安装库）

```bash
pip install markdown reportlab
python3 convert_to_pdf.py
```

## 方法 5: 使用 Chrome/Edge 浏览器

1. 使用 Markdown 预览扩展打开文件
2. 打印 → 保存为 PDF

---

**推荐**: 方法 1 (Pandoc) 提供最好的格式控制和质量。
