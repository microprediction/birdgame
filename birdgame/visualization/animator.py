import typing

from matplotlib import pyplot as plt
from tqdm.auto import tqdm


class AnimatorFunctions:

    def setup(self):
        pass

    def show(self, figure):
        from IPython.display import clear_output
        clear_output(wait=True)
        self._display(figure)

    def cleanup(self):
        pass

    def _display(self, figure):
        from IPython.display import display
        display(figure)


class ColabAnimatorFunctions(AnimatorFunctions):
    def setup(self):
        self._javascript("""
        const outputBody = document.getElementById('output-body');
        outputBody.style.position = "relative";
        console.log({ outputBody })
        """)

    def show(self, figure):
        self._display(figure)

        self._javascript("""
        const images = [...document.querySelectorAll('#output-area img, .output_subarea img')];
        console.log({ images: JSON.stringify(images.map((x) => x.parentNode.parentNode.classList[1])) })
        const plotImg = images[images.length - 1].parentNode;
        console.log({ plotImg: plotImg.parentNode.classList[1] })

        plotImg.style.position = "absolute";
        plotImg.style.top = "32px"; /* for tqdm */
        plotImg.style.left = "0px";
        plotImg.style.width = "100%";
        plotImg.style.height = "100%";

        const previousImages = images.slice(0, images.length - 1);
        console.log({ previousImages: JSON.stringify(previousImages.map((x) => x.parentNode.parentNode.classList[1])) })
        for (const previous of previousImages) {
            /* imageContainer . outputContainer */
            previous.parentNode.parentNode.remove();
        }
        """)

    def cleanup(self):
        self._javascript("""
        const images = [...document.querySelectorAll('#output-area img, .output_subarea img')];
        for (const image of images) {
            const plotImg = image.parentNode;
            plotImg.style.position = null;
            plotImg.style.top = null;
            plotImg.style.left = null;
            plotImg.style.width = null;
            plotImg.style.height = null;
        }

        const previousImages = images.slice(0, images.length - 1);
        for (const previous of previousImages) {
            /* imageContainer . outputContainer */
            previous.parentNode.parentNode.remove();
        }

        const outputBody = document.getElementById('output-body');
        outputBody.style.position = null;
        """)

    def _javascript(self, code: str):
        from IPython.display import Javascript
        self._display(Javascript(code))


def animate(
    iterator: typing.Any,
    animate: typing.Callable[[typing.Any], None],
    interval=100,
    environment: typing.Literal['auto', 'generic' 'colab'] = 'auto',
):
    if environment == 'auto':
        from IPython import get_ipython

        ipython = get_ipython()
        if ipython is not None and 'google.colab' in str(ipython):
            environment = 'colab'
        else:
            environment = 'generic'

    if environment == 'colab':
        functions = ColabAnimatorFunctions()
    elif environment == 'generic':
        functions = AnimatorFunctions()
    else:
        raise ValueError(f"Unknown environment: {environment}")

    interval /= 1000

    with plt.ioff():
        try:
            functions.setup()

            for value in tqdm(iterator):
                figure = animate(value)
                if figure is None:
                    continue

                functions.show(figure)

                plt.pause(interval)
        finally:
            functions.cleanup()
