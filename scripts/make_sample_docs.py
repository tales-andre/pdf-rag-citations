"""Generate the fictional sample corpus in sample_docs/.

Acme Supply Co. is a made-up company. The PDFs are small on purpose, but each
has several pages so page-level citations can be checked.

Run: python scripts/make_sample_docs.py
"""

from __future__ import annotations

from pathlib import Path

from reportlab import rl_config
from reportlab.lib.pagesizes import LETTER
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import PageBreak, Paragraph, SimpleDocTemplate, Spacer

rl_config.invariant = 1  # reproducible bytes (no timestamps)

OUT = Path(__file__).resolve().parent.parent / "sample_docs"

# Each document: (file name, title, pages). Each page: list of (kind, text),
# where kind is "h" for a heading or "p" for a paragraph.
DOCS = [
    (
        "acme-ergodesk-pro-product-guide.pdf",
        "ErgoDesk Pro Standing Desk - Product Guide",
        [
            [
                ("h", "Overview"),
                ("p", "The ErgoDesk Pro is an electric sit-stand desk sold by Acme Supply Co. "
                      "It uses two motors, one in each leg, to move the desktop smoothly."),
                ("h", "Specifications"),
                ("p", "The height range is 62 cm to 127 cm, measured from the floor to the top of the desktop. "
                      "The maximum load capacity is 120 kg, evenly distributed. "
                      "Desktops are available in three sizes: 120 x 60 cm, 140 x 70 cm and 160 x 80 cm. "
                      "The lifting speed is 38 mm per second."),
                ("p", "The control panel has an up button, a down button, a memory button marked M and four preset buttons numbered 1 to 4."),
            ],
            [
                ("h", "Assembly"),
                ("p", "Assembly takes about 45 minutes. Two people are recommended because the frame is heavy. "
                      "The only tool needed is a Phillips screwdriver; all other tools are included in the box."),
                ("p", "Attach the legs to the crossbar first, then fix the frame to the underside of the desktop. "
                      "Connect both motor cables and the control panel cable to the control box before plugging in the power cord."),
                ("h", "Memory presets"),
                ("p", "To save a memory preset, move the desk to the height you want, press M, then press and hold a preset button from 1 to 4 for three seconds. "
                      "The display blinks twice to confirm the preset was saved. "
                      "To use a preset, press its number once and the desk moves to the saved height."),
            ],
            [
                ("h", "Troubleshooting"),
                ("p", "Error code E01 means motor overload. Remove some weight from the desktop, then reset the desk by holding the down button for 10 seconds until it beeps."),
                ("p", "Error code E02 means the two legs are out of balance. Lower the desk fully and hold the down button for 10 seconds to recalibrate."),
                ("p", "If the display stays blank, check that the power cord is firmly connected to the control box and to the wall outlet."),
                ("h", "Care and cleaning"),
                ("p", "Clean the desktop with a soft damp cloth. Do not use solvents or abrasive cleaners, because they damage the surface finish."),
            ],
        ],
    ),
    (
        "acme-returns-and-refunds-sop.pdf",
        "SOP-RET-01 Returns and Refunds",
        [
            [
                ("h", "Purpose and scope"),
                ("p", "This procedure tells support agents how to handle customer returns and refunds for all products sold online by Acme Supply Co."),
                ("h", "Return window"),
                ("p", "Customers can return unopened items within 30 days of delivery for a full refund. "
                      "Opened items can be returned within 14 days of delivery and are subject to a 15% restocking fee. "
                      "The return window is counted from the delivery date shown in the carrier tracking."),
            ],
            [
                ("h", "Procedure"),
                ("p", "Step 1. Confirm the order number and the delivery date in the order system."),
                ("p", "Step 2. Create a return authorization (RMA) number in the order system and send it to the customer."),
                ("p", "Step 3. The customer receives a prepaid return label by email within one business day."),
                ("p", "Step 4. When the item reaches the warehouse, it is inspected. "
                      "Refunds are issued to the original payment method within 5 business days after the warehouse inspection."),
                ("p", "Return shipping is free for unopened items. For opened items, the cost of the return label is deducted from the refund."),
            ],
            [
                ("h", "Exceptions"),
                ("p", "Custom-cut desktops and clearance items are final sale and cannot be returned."),
                ("p", "Items damaged on arrival must be reported within 48 hours of delivery, with photos of the item and of the box."),
                ("p", "Refunds over $500 require approval from a support manager before they are issued."),
            ],
        ],
    ),
    (
        "acme-shipping-faq.pdf",
        "Shipping and Delivery FAQ",
        [
            [
                ("h", "How long does shipping take?"),
                ("p", "Standard shipping takes 3 to 7 business days. Expedited shipping arrives in 2 business days and costs $24.95."),
                ("h", "Is shipping free?"),
                ("p", "Standard shipping is free for orders over $75. Orders below $75 pay a flat rate of $8.95."),
                ("h", "Where do you ship?"),
                ("p", "We ship to the 48 contiguous United States only. We do not ship furniture to PO boxes."),
            ],
            [
                ("h", "How do I track my order?"),
                ("p", "You receive an email with a tracking link within 24 hours after your order leaves the warehouse."),
                ("h", "What is white-glove delivery?"),
                ("p", "White-glove delivery is available for furniture for $149. "
                      "The crew brings the item to the room you choose, assembles it and removes all packaging."),
                ("h", "What if I miss the delivery?"),
                ("p", "The carrier tries to deliver twice. After the second failed attempt, the package goes back to the warehouse and support contacts you to reschedule."),
            ],
        ],
    ),
    (
        "acme-warranty-policy.pdf",
        "Warranty Policy",
        [
            [
                ("h", "Coverage periods"),
                ("p", "The ErgoDesk Pro frame is covered for 10 years. "
                      "Motors and electronics, including the control box and control panel, are covered for 5 years. "
                      "The desktop surface is covered for 2 years. "
                      "Office chairs sold by Acme Supply Co. are covered for 7 years."),
                ("p", "Warranty coverage requires proof of purchase, such as the order confirmation email."),
            ],
            [
                ("h", "What is not covered"),
                ("p", "The warranty does not cover damage from misuse, from loads above the rated capacity, or normal cosmetic wear such as scratches."),
                ("h", "How to make a warranty claim"),
                ("p", "Fill in the warranty claim form and include the serial number. "
                      "The serial number is printed on a label under the desktop, next to the control box. "
                      "Approved replacement parts are shipped free of charge."),
            ],
        ],
    ),
    (
        "acme-support-escalation-sop.pdf",
        "SOP-SUP-03 Support Escalation",
        [
            [
                ("h", "Tier 1 responsibilities"),
                ("p", "Tier 1 agents handle order status questions, returns, and basic troubleshooting using the product guides."),
                ("h", "Response time targets"),
                ("p", "Tier 1 answers emails within 1 business day and live chats within 2 minutes."),
            ],
            [
                ("h", "When to escalate to Tier 2"),
                ("p", "Escalate when an error code persists after the customer has reset the desk. "
                      "Escalate any refund over $500, because it needs manager approval."),
                ("h", "Safety concerns"),
                ("p", "If a customer reports smoke or a burning smell, tell the customer to unplug the desk immediately and stop using it, then escalate the ticket to Tier 2 as urgent."),
                ("h", "How to escalate"),
                ("p", "Add the tag T2-escalate to the ticket and write a short summary of the steps already tried. "
                      "Tier 2 responds within 4 business hours."),
            ],
        ],
    ),
]


def build(path: Path, title: str, pages: list[list[tuple[str, str]]]) -> None:
    styles = getSampleStyleSheet()
    body = ParagraphStyle("body", parent=styles["BodyText"], fontSize=11, leading=15, spaceAfter=8)
    heading = ParagraphStyle("h", parent=styles["Heading2"], fontSize=13, spaceBefore=6, spaceAfter=4)
    title_style = ParagraphStyle("t", parent=styles["Title"], fontSize=18)
    note = ParagraphStyle("note", parent=styles["Italic"], fontSize=9, textColor="#555555")

    def footer(canvas, doc):
        canvas.saveState()
        canvas.setFont("Helvetica", 8)
        canvas.drawString(inch, 0.6 * inch, f"Acme Supply Co. (fictional company) - {title}")
        canvas.drawRightString(LETTER[0] - inch, 0.6 * inch, f"Page {doc.page}")
        canvas.restoreState()

    story = [Paragraph(title, title_style),
             Paragraph("Fictional sample document created for an open-source demo.", note),
             Spacer(1, 10)]
    for i, page in enumerate(pages):
        if i:
            story.append(PageBreak())
        for kind, text in page:
            story.append(Paragraph(text, heading if kind == "h" else body))

    doc = SimpleDocTemplate(str(path), pagesize=LETTER, title=title, author="Acme Supply Co. (fictional)",
                            leftMargin=inch, rightMargin=inch, topMargin=inch, bottomMargin=inch)
    doc.build(story, onFirstPage=footer, onLaterPages=footer)


def main() -> None:
    OUT.mkdir(exist_ok=True)
    for name, title, pages in DOCS:
        build(OUT / name, title, pages)
        print(f"wrote {OUT / name} ({len(pages)} pages)")


if __name__ == "__main__":
    main()
