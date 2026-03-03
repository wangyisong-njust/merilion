from markdown_pdf import MarkdownPdf, Section
import sys
import pathlib
import re

def main():
    if len(sys.argv) != 3:
        print("Usage: python convert_md_to_pdf.py <input_markdown_file> <output_pdf_file>")
        sys.exit(1)

    input_file = pathlib.Path(sys.argv[1])
    output_file = pathlib.Path(sys.argv[2])

    if not input_file.exists():
        print(f"Error: Input file '{input_file}' does not exist.")
        sys.exit(1)

    # Read the Markdown content
    with input_file.open("r", encoding="utf-8") as f:
        markdown_content = f.read()

    # Create a MarkdownPdf instance
    pdf = MarkdownPdf(toc_level=3, optimize=True)

    # Add a section with the Markdown content
    def extract_sections(content, level=3):
        pattern = re.compile(r"^(#{1,%d})\s+(.*)" % level, re.MULTILINE)
        sections = []
        last_pos = 0
        last_level = 0
        for match in pattern.finditer(content):
            current_level = len(match.group(1))
            title = match.group(2).strip()
            start_pos = match.start()
            if last_pos > 0:
                sections.append(Section(content[last_pos:start_pos].strip())) # , user_css=f"h{current_level}" + " {color: blue;}"
            last_pos = start_pos
            last_level = current_level
        if last_pos < len(content):
            sections.append(Section(content[last_pos:].strip())) # , user_css=f"h{last_level}" + " {color: blue;}"
        return sections

    sections = extract_sections(markdown_content, level=3)
    for section in sections:
        pdf.add_section(section)

    # pdf.add_section(Section("# Title\n\nThis is some content."))
    # pdf.add_section(Section("## Subtitle\n\nMore content here.", user_css="h2 {color: blue;}"))
    # Save the PDF to the specified output file
    pdf.save(output_file)
    print(f"PDF successfully created: {output_file}")

if __name__ == "__main__":
    main()