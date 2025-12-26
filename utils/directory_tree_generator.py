import pathlib
import io
import humanize
from rich.tree import Tree
from rich.text import Text
from rich.console import Console

class DirectoryTreeGenerator:
    """
    一个用于生成指定目录的 ASCII 树状结构图的工具类。
    旨在为 LLM 提供清晰的文件系统上下文，包含文件大小和目录项统计等元数据。
    """

    def __init__(self, root_path: str, max_depth: int = None, ignore_patterns: list = None, 
                 collapse_extensions: list = None, collapse_threshold: int = 3, preview_files: list = None):
        """
        初始化目录树生成器。

        Args:
            root_path (str): 需要扫描的根目录路径。
            max_depth (int, optional): 遍历的最大深度。默认为 None (无限制)。
            ignore_patterns (list, optional): 需要忽略的文件名模式列表。
            collapse_extensions (list, optional): 需要检查折叠的文件后缀列表（不带点，例如 ['jpg', 'png']）。
                                                  默认为常见的图像、视频、数据文件格式。
            collapse_threshold (int, optional): 触发折叠的文件数量阈值。默认为 3。
            preview_files (list, optional): 需要显示预览内容的特定文件名列表。默认为 ['sample_submission.csv', 'train.csv']。
        """
        self.root_path = pathlib.Path(root_path)
        self.max_depth = max_depth
        self.ignore_patterns = ignore_patterns or ['.git', '__pycache__', '.DS_Store']
        
        # 默认的折叠后缀列表
        default_collapse = [
            # Images
            'jpg', 'jpeg', 'png', 'gif', 'bmp', 'tiff', 'webp', 'svg', 'ico',
            # Videos
            'mp4', 'mkv', 'avi', 'mov', 'wmv', 'flv', 'webm',
            # Data / Config
            'json', 'csv', 'xml', 'yaml', 'yml', 'toml', 'ini',
            # Archives
            'zip', 'tar', 'gz', 'rar', '7z',
            # Documents
            'pdf', 'doc', 'docx', 'xls', 'xlsx', 'ppt', 'pptx', 'txt', 'md',
            # Others
            'HEIC'
        ]
        self.collapse_extensions = collapse_extensions if collapse_extensions is not None else default_collapse
        # normalize to lowercase and ensure no leading dots
        self.collapse_extensions = [ext.lower().lstrip('.') for ext in self.collapse_extensions]
        
        self.collapse_threshold = collapse_threshold
        self.preview_files = preview_files if preview_files is not None else ['sample_submission.csv', 'train.csv']
        
        if not self.root_path.exists():
            raise FileNotFoundError(f"Path not found: {root_path}")
        if not self.root_path.is_dir():
             raise NotADirectoryError(f"Path is not a directory: {root_path}")

    def generate(self) -> tuple[str, dict]:
        """
        生成 Markdown 格式的目录树字符串和文件预览字典。

        Returns:
            tuple[str, dict]: (包含 ASCII 树的 Markdown 代码块, 文件预览字典 {abs_path: content})
        """
        self.collected_previews = {}
        root_label = self._format_node_label(self.root_path, is_root=True)
        tree = Tree(root_label)

        self._build_tree(self.root_path, tree, current_depth=0)

        return self._render_tree_to_string(tree), self.collected_previews

    def _get_file_preview(self, file_path: pathlib.Path) -> str:
        """
        获取文件内容预览。
        如果是 md 文件读取前 10 行，如果是 csv 文件读取前 5 行。
        """
        try:
            lines_to_read = 5
            if file_path.suffix.lower() == '.md':
                lines_to_read = 10
            
            preview_lines = []
            with open(file_path, 'r', encoding='utf-8') as f:
                for _ in range(lines_to_read):
                    line = f.readline()
                    if not line:
                        break
                    preview_lines.append(line.rstrip())
            
            if not preview_lines:
                return "(Empty file)"
                
            return "\n".join(preview_lines) + "\n..."
            
        except UnicodeDecodeError:
            return "(Preview unavailable: Binary file or encoding error)"
        except Exception as e:
            return f"(Preview unavailable: {str(e)})"

    def _build_tree(self, current_path: pathlib.Path, tree_node: Tree, current_depth: int):
        """
        递归构建 rich Tree，包含折叠逻辑。
        """
        if self.max_depth is not None and current_depth >= self.max_depth:
            tree_node.add("... (max depth reached)")
            return

        try:
            # 获取所有子项
            all_children = [p for p in current_path.iterdir() if p.name not in self.ignore_patterns]
        except PermissionError:
            tree_node.add("[red]<Permission Denied>[/red]")
            return

        # 分离目录和文件
        dirs = sorted([p for p in all_children if p.is_dir()], key=lambda p: p.name.lower())
        files = sorted([p for p in all_children if p.is_file()], key=lambda p: p.name.lower())

        # 1. 处理目录 (递归)
        for d in dirs:
            label = self._format_node_label(d)
            subdir_node = tree_node.add(label, style="bold blue")
            self._build_tree(d, subdir_node, current_depth + 1)

        # 2. 统计文件后缀，决定哪些需要折叠
        # extension_counts: { 'jpg': [file_path1, file_path2, ...], ... }
        extension_groups = {}
        for f in files:
            # 获取后缀 (不带点，转小写)
            ext = f.suffix.lstrip('.').lower()
            if ext in self.collapse_extensions:
                if ext not in extension_groups:
                    extension_groups[ext] = []
                extension_groups[ext].append(f)

        # 确定哪些后缀需要折叠
        collapsed_extensions = set()
        for ext, file_list in extension_groups.items():
            if len(file_list) > self.collapse_threshold:
                collapsed_extensions.add(ext)

        # 3. 添加折叠的汇总节点
        # 对已折叠的后缀按字母排序显示
        for ext in sorted(collapsed_extensions):
            count = len(extension_groups[ext])
            # 显示汇总信息，例如 "📦 *.jpg (15 files)"
            # 也可以选择计算总大小，这里暂时只显示数量
            tree_node.add(f"📦 *.{ext} ({count} files)")

        # 4. 添加未折叠的具体文件
        for f in files:
            ext = f.suffix.lstrip('.').lower()
            # 如果该文件的后缀被折叠了，则跳过
            if ext in collapsed_extensions:
                continue
            
            # 否则正常显示
            label = self._format_node_label(f)
            tree_node.add(label)
            
            # 检查是否需要预览
            if f.name in self.preview_files:
                preview_content = self._get_file_preview(f)
                # 使用绝对路径作为 Key
                self.collected_previews[str(f.resolve())] = preview_content

    def _format_node_label(self, path: pathlib.Path, is_root: bool = False):
        """
        格式化节点显示的文本，包含元数据。
        """
        name = path.name
        if is_root:
            name = f"{path.resolve()}/"

        meta_info = ""
        
        if path.is_dir():
            # 统计目录下的文件数（浅层统计，不递归，为了性能）
            try:
                # 再次过滤，保持统计的一致性
                items = [p for p in path.iterdir() if p.name not in self.ignore_patterns]
                count = len(items)
                meta_info = f" ({count} items)"
            except PermissionError:
                meta_info = " (Access Denied)"
        else:
            # 文件显示大小
            try:
                size = path.stat().st_size
                human_size = humanize.naturalsize(size, binary=True) # binary=True 使用 KiB, MiB
                meta_info = f" ({human_size})"
            except OSError:
                 meta_info = " (Unknown size)"

        # 构建 Text 对象以便利用 rich 的格式化能力 (虽然最后会转纯文本，但 rich 可以处理 emoji 等)
        # 这里我们简单返回字符串供 Tree 使用，让 Tree 处理结构
        
        if path.is_dir():
             return f"📂 {name}{meta_info}"
        else:
             return f"📄 {name}{meta_info}"

    def _render_tree_to_string(self, tree: Tree) -> str:
        """
        使用 Console 将 rich Tree 渲染为字符串。
        """
        console = Console(file=io.StringIO(), width=100, force_terminal=False, color_system=None)
        console.print(tree)
        output = console.file.getvalue()
        
        # 封装在 Markdown 代码块中
        markdown_output = f"```\n{output}```"
        return markdown_output
