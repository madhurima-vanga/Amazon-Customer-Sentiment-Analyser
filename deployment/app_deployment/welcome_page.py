import streamlit as st

def main():
    st.title("🎉 Welcome to Your E-commerce Superpower! 🚀")
    st.write(
        """
        Hey there, superstar seller! 👋 Ready to level up your e-commerce game? 

        With our **Sentiment Analysis Tool**, you can:
        - 🕵️‍♂️ Spy on customer sentiment and uncover what they *really* think about your products. 
        - 📊 Dive into insights, performance stats, and trends that help you stand out in your category.
        - 🤼‍♀️ Go head-to-head with competitors by comparing your product's performance with theirs.
        - 🏆 Use category benchmarks to stay ahead of the game and always be the customer's top choice.

        But wait, there's more! Our cutting-edge AI and MLOps tech ensure you have the smoothest, most reliable experience ever. It's like having a personal data scientist on call 24/7. 😎

        So, what are you waiting for? Let’s transform your customer reviews into your secret weapon for success!
        """
    )

    if st.button("✨ Try It Out! ✨"):
        st.session_state.page = "Main"
