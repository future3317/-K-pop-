# KPopScope 课程论文 LaTeX 初稿

本文件夹包含一份中文 LaTeX 论文初稿，可作为《音频信息处理》大作业报告基础版：

- `main.tex`：中文论文主文件
- `references.bib`：参考文献
- `README.md`：使用说明

## 编译方式

推荐使用 XeLaTeX：

```bash
latexmk -xelatex main.tex
```

或手动：

```bash
xelatex main.tex
bibtex main
xelatex main.tex
xelatex main.tex
```

## 后续必须替换的内容

文中所有 `待替换：...` 都需要在 Codex 跑完项目实验后填入真实结果，尤其是：

1. 作者、课程、教师信息
2. 实验环境
3. 分类指标表
4. stem contribution case study
5. 段落结构示例
6. 人工评测结果
7. 具体歌曲名和时间戳

注意：当前实验表中的数值位置是占位模板，不能当作真实实验结果提交。
