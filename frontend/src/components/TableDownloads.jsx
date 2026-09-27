import { jsPDF } from "jspdf";
import autoTable from "jspdf-autotable";

function downloadBlob(blob, filename) {
  const url = URL.createObjectURL(blob);
  const link = document.createElement("a");
  link.href = url;
  link.download = filename;
  link.click();
  URL.revokeObjectURL(url);
}

function csvCell(value) {
  return `"${String(value ?? "").replaceAll('"', '""')}"`;
}

function pdfCell(value) {
  return String(value ?? "")
    .normalize("NFKD")
    .replace(/[\u0300-\u036f]/g, "")
    .replace(/[\u2010-\u2015]/g, "-")
    .replace(/[\u2018\u2019]/g, "'")
    .replace(/[\u201c\u201d]/g, '"')
    .replace(/\u2192/g, "to")
    .replace(/\p{Extended_Pictographic}/gu, "")
    .replace(/\uFE0F/g, "")
    .replace(/[^\x20-\x7e\xa0-\xff]/g, " ");
}

function downloadCsv(table, filename) {
  const { columns, rows } = table;
  const csv = [
    columns.map(csvCell).join(","),
    ...rows.map((row) => columns.map((column) => csvCell(row[column])).join(",")),
  ].join("\r\n");
  downloadBlob(new Blob(["\uFEFF", csv], { type: "text/csv;charset=utf-8" }), `${filename}.csv`);
}

function downloadPdf(table, title, subtitle, filename) {
  const { columns, rows } = table;
  const wideTable = columns.length > 8;
  const doc = new jsPDF({
    orientation: wideTable ? "landscape" : "portrait",
    unit: "pt",
    format: wideTable ? "a3" : "a4",
  });
  const pageWidth = doc.internal.pageSize.getWidth();
  const pageHeight = doc.internal.pageSize.getHeight();

  autoTable(doc, {
    head: [columns.map(pdfCell)],
    body: rows.map((row) => columns.map((column) => pdfCell(row[column]))),
    startY: 66,
    margin: { top: 66, right: 28, bottom: 34, left: 28 },
    styles: {
      font: "helvetica",
      fontSize: wideTable ? 6.5 : 8,
      cellPadding: 3,
      overflow: "linebreak",
      valign: "middle",
    },
    headStyles: { fillColor: [43, 124, 184], textColor: 255, fontStyle: "bold" },
    alternateRowStyles: { fillColor: [239, 245, 251] },
    horizontalPageBreak: wideTable,
    horizontalPageBreakRepeat: wideTable ? [0, 1] : undefined,
    didDrawPage: () => {
      doc.setFont("helvetica", "bold");
      doc.setFontSize(14);
      doc.text(pdfCell(title), 28, 28);
      doc.setFont("helvetica", "normal");
      doc.setFontSize(8);
      doc.text(pdfCell(subtitle), 28, 43);
      doc.text(`Page ${doc.internal.getCurrentPageInfo().pageNumber}`, pageWidth - 28, pageHeight - 14, {
        align: "right",
      });
    },
  });
  doc.save(`${filename}.pdf`);
}

export default function TableDownloads({ table, title, subtitle, filename }) {
  if (!table?.columns?.length || !table.rows?.length) return null;

  return (
    <div className="table-downloads" aria-label={`${title} downloads`}>
      <button type="button" onClick={() => downloadCsv(table, filename)}>
        Download CSV
      </button>
      <button type="button" onClick={() => downloadPdf(table, title, subtitle, filename)}>
        Download PDF
      </button>
    </div>
  );
}
