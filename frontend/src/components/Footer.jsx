/** Betting Oracle footer — byte-for-byte the markup in ``footer.py``. */
export function Footer() {
  return (
    <div
      className="betting-oracle-footer"
      data-testid="stMarkdownContainer"
      data-footer="betting-oracle"
      dangerouslySetInnerHTML={{
        __html: `
<div style="text-align: center; padding: 20px 0; border-top: 1px solid #e0e0e0; margin-top: 40px;">
    <p style="margin: 0 0 10px 0; font-size: 14px; color: #666; font-family: sans-serif;">
        Powered by <a href="https://www.betting-oracle.com" target="_blank" style="color: #3b82f6; text-decoration: none; font-weight: bold;">Betting Oracle</a>
    </p>
    <p style="margin: 0 0 15px 0; font-size: 12px; color: #888; font-family: sans-serif;">
        Sports Prediction Analytics
    </p>
    <a href="https://www.betting-oracle.com" target="_blank">
        <img src="https://raw.githubusercontent.com/gmalbert/betting-oracle/main/data_files/logo.png"
             alt="Betting Oracle Logo"
             style="height: 60px; width: auto; border: none;">
    </a>
</div>
`.trim(),
      }}
    />
  );
}

export default Footer;
